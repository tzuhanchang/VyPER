import os
import sys
import math
import h5py
import pickle
import base64

from tqdm import tqdm
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from VyPER.data import VyPERDataset


class PreprocessingWrite(IterableDataset):
    r"""Preprocessing graphs and write them into a HDF5 database.

    Args:
        root (str): path to the dataset.
        config (str): path to the configuration YAML file.
    """
    def __init__(self, root: str, config: str, training: bool=True):
        super(PreprocessingWrite).__init__()
        self.root = root
        self.config = config

        # Load fondation dataset
        self.master = VyPERDataset(root, config, training=training)
        self.size = len(self.master)

    def create_db(self, db_path: str, num_events: int) -> None:
        f  = h5py.File(db_path, 'a')
        db = f.create_dataset('Graphs', shape=(num_events,), dtype=h5py.string_dtype(encoding='utf-8'))
        return f, db

    def write_graph(self, db, index: int, G: Data) -> None:
        graph_bytes = pickle.dumps(G, protocol=pickle.HIGHEST_PROTOCOL)
        db[index] = base64.b64encode(graph_bytes)

    def __iter__(self):
        # Worker info
        worker_info = get_worker_info()

        # Workload
        if worker_info is None:
            iter_start = 0
            iter_end = self.size
        else:
            per_worker = int(math.ceil((self.size) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.size)

        # Create a database for each worker
        if worker_info is None:
            db_path = os.path.splitext(self.root)[0] + '-0' + '.db'
        else:
            db_path = os.path.splitext(self.root)[0] + f'-{worker_info.id}' + '.db'
        f, db = self.create_db(db_path, int(iter_end-iter_start))

        for idx in tqdm(range(iter_start, iter_end),
                        desc=f"Worker {worker_id if worker_info else 'main'}",
                        total=iter_end-iter_start,
                        position=worker_id if worker_info is not None else 0,
                        dynamic_ncols=False, ncols=100, nrows=4, file=sys.stderr,
                        leave=True, unit='evt', miniters=100, ascii=True):

            G = self.master[idx]
            self.write_graph(db, idx-iter_start, G)
            yield G

        f.close()


class GraphDB():
    r"""VyPER graph database. Creates a HDF5 file to store processed graphs as 
    bytes objects, allowing fast graph processing and loading.

    Args:
        root (str): path to the dataset.
        config (str): path to the configuration YAML file.
        num_workers (optional, int): number of workers used for preprocessing. 
            (default: `int=0`)
        batch_size (optional, int): number of graphs in each preprocessing batch. 
            (default: `int=128`)
        force_reload (optional, bool): force to recreate a new database 
            if one already exisits. (default: `bool=False`)
    """
    def __init__(self, root: str, config: str, training: bool=True, num_workers: int=0,
                 batch_size: int=128, force_reload: bool=False):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_vars = None

        self.db_path = os.path.splitext(root)[0] + '.db'

        if force_reload and os.path.exists(self.db_path):
            os.remove(self.db_path)

        if os.path.exists(self.db_path) is False:
            print(f"Running graph pre-processing on dataset '{root}' with {num_workers} workers:")
            self.batch_pw = PreprocessingWrite(root=root, config=config, training=training)
            self.size = self.batch_pw.size
            self.dataset_vars = vars(self.batch_pw.master)
            self.batch_files = [os.path.splitext(root)[0] + '-' + str(i) + '.db' for i in range(num_workers)] \
                if num_workers > 0 else [os.path.splitext(root)[0] + '-0' + '.db']
            self.cleanup()  # Remove worker files created in a failed run
            self.preprocessing_write()

            self.concatenate()
            self.cleanup()  # Clean up again
            print(f"\nGraph database {self.db_path} created.")
        else:
            master = VyPERDataset(root, config, training=training)
            self.dataset_vars = vars(master)

        self.db = h5py.File(self.db_path, 'r')['Graphs']
        self.size = len(self.db)

    def preprocessing_write(self) -> None:
        loader = DataLoader(self.batch_pw,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                            shuffle=False,
                            persistent_workers=True,
                            drop_last=False)
        for data in loader:
            pass

    def concatenate(self) -> None:
        current_index = 0
        with h5py.File(self.db_path, 'a') as output_f:
            # Create the Graphs dataset with total size
            output_db = output_f.create_dataset(
                'Graphs', 
                shape=(self.size,), 
                dtype=h5py.string_dtype(encoding='utf-8')
            )

            for input_file in self.batch_files:
                with h5py.File(input_file, 'r') as input_f:
                    input_db = input_f['Graphs']
                    num_events = len(input_db)
                    
                    output_db[current_index:current_index + num_events] = input_db[:]
                    current_index += num_events

    def cleanup(self) -> None:
        for file in self.batch_files:
            if os.path.exists(file):
                os.remove(file)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index) -> Data:
        graph_bytes = base64.b64decode(self.db[index])
        return pickle.loads(graph_bytes, encoding='utf-8')


class VyPEROnDiskDataset(Dataset):
    def __init__(self, root: str, config: str, training: bool=True, force_reload: bool=False,
                 batch_size: int=128, num_workers: int=0) -> None:
        self.root = root
        self.config = config
        self.force_reload = force_reload
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train_mode = training

        self.db = GraphDB(root=root, config=config, num_workers=num_workers,
                          batch_size=batch_size, force_reload=force_reload)

        for key, value in self.db.dataset_vars.items():
            self.__setattr__(key, value)

        super().__init__(root, transform=None, pre_transform=None,
                         pre_filter=None)

    def __getitem__(self, index) -> Data:
        return self.db[index]

    def __len__(self):
        return self.size
