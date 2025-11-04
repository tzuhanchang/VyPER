import torch

from rich import get_console
from rich.table import Table
from lightning import LightningDataModule
from torch.utils.data import Subset, random_split
from torch_geometric.loader import DataLoader
from typing import Optional

from VyPER.data import VyPERDataset


class VyPERDataModule(LightningDataModule):
    def __init__(
        self,
        config: str,
        train_set: Optional[str]=None,
        val_set: Optional[str]=None,
        predict_set: Optional[str]=None,
        event_filter: Optional[str]=None,
        cache_dir: Optional[str]=None,
        force_reload: bool=False,
        batch_size: int=128,
        percent_train_samples: float=0.9,
        drop_last: bool=False,
        num_workers: int=0,
        pin_memory: bool=False
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.train_set = train_set
        self.val_set = val_set
        self.predict_set = predict_set
        self.event_filter = event_filter
        self.cache_dir = cache_dir
        self.force_reload = force_reload
        self.batch_size = batch_size
        self.percent_train_samples = percent_train_samples
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_data = None
        self.val_data = None
        self.predict_data = None

        self.node_in_channels = None
        self.edge_in_channels = None
        self.glob_in_channels = None
        self.nu_out_channels  = None
        self.nu_reverse_transform_methods = None
        self.neutrino_momentum_func = None
        self.neutrino_4vector_func = None
        self.neutrino_4vector_loc = None
        self._use_hyperedge = False
        self._use_diffusion = False

    def setup(self, stage: str) -> None:
        if self.train_set is not None:
            if self.val_set is None or self.train_set == self.val_set:
                if self.event_filter is not None:
                    raise NotImplementedError("`event_filter` is currently not supported when"+\
                                              " `val_set` is not provided.")

                print("Creating validation set using "
                    +f"{round((1-self.percent_train_samples)*100,2)}% of the file.")

                data = VyPERDataset(root=self.train_set, config=self.config, training=True,
                                    cache_dir=self.cache_dir, force_reload=self.force_reload)

                self._use_diffusion = data._use_diffusion
                self._use_hyperedge = data._use_hyperedge
                self.node_in_channels = data.node_in_channels
                self.edge_in_channels = data.edge_in_channels
                self.glob_in_channels = data.glob_in_channels
                self.nu_out_channels  = data.nu_out_channels
                self.nu_reverse_transform_methods = data.nu_reverse_transform_methods
                self.neutrino_momentum_func = data.neutrino_momentum_func
                self.neutrino_4vector_func = data.neutrino_4vector_func
                self.neutrino_4vector_loc = data.neutrino_4vector_loc

                self.train_data, self.val_data = random_split(
                    data, 
                    [self.percent_train_samples, 1-self.percent_train_samples])
                del data
            else:
                self.train_data = VyPERDataset(root=self.train_set, config=self.config, training=True,
                                               cache_dir=self.cache_dir, force_reload=self.force_reload)
                self.val_data = VyPERDataset(root=self.val_set, config=self.config, training=True,
                                             cache_dir=self.cache_dir, force_reload=self.force_reload)

                self._use_diffusion = self.train_data._use_diffusion
                self._use_hyperedge = self.train_data._use_hyperedge
                if self.node_in_channels is None:
                    self.node_in_channels = self.train_data.node_in_channels
                    self.edge_in_channels = self.train_data.edge_in_channels
                    self.glob_in_channels = self.train_data.glob_in_channels
                    self.nu_out_channels  = self.train_data.nu_out_channels
                    self.nu_reverse_transform_methods = self.train_data.nu_reverse_transform_methods
                    self.neutrino_momentum_func = self.train_data.neutrino_momentum_func
                    self.neutrino_4vector_func = self.train_data.neutrino_4vector_func
                    self.neutrino_4vector_loc = self.train_data.neutrino_4vector_loc

                if self.event_filter is not None:
                    self.train_data = Subset(
                        self.train_data,
                        torch.argwhere(getattr(self.train_data, self.event_filter)==1).flatten()
                    )
                    self.val_data = Subset(
                        self.val_data,
                        torch.argwhere(getattr(self.val_data, self.event_filter)==1).flatten()
                    )

        if self.predict_set is not None:
            self.predict_data = VyPERDataset(root=self.predict_set, config=self.config, training=False,
                                             cache_dir=self.cache_dir, force_reload=self.force_reload)

            self._use_diffusion = self.predict_data._use_diffusion
            self._use_hyperedge = self.predict_data._use_hyperedge
            if self.node_in_channels is None:
                self.node_in_channels = self.predict_data.node_in_channels
                self.edge_in_channels = self.predict_data.edge_in_channels
                self.glob_in_channels = self.predict_data.glob_in_channels
                self.nu_out_channels  = self.predict_data.nu_out_channels
                self.nu_reverse_transform_methods = self.predict_data.nu_reverse_transform_methods
                self.neutrino_momentum_func = self.predict_data.neutrino_momentum_func
                self.neutrino_4vector_func = self.predict_data.neutrino_4vector_func
                self.neutrino_4vector_loc = self.predict_data.neutrino_4vector_loc

        if self.train_data is None and self.val_data is None and self.predict_data is None:
            raise RuntimeError("No datasets have been provided. Abort!")

        # Print out stats.
        console = get_console()
        table = Table(title="Dataset Status",header_style="orange1")
        table.add_column("Name", justify="left")
        table.add_column("Value", justify="left")
        table.add_row("Drop last batch", str(self.drop_last))
        table.add_row("Force reload", str(self.force_reload))
        table.add_row("Cache directory", str(self.cache_dir))
        if self.train_data is not None:
            table.add_row("Training file path", str(self.train_set))
            table.add_row("Training samples", str(len(self.train_data)))
        if self.val_data is not None:
            if self.val_set is not None:
                table.add_row("Validation file path", str(self.val_set))
            table.add_row("Validation samples", str(len(self.val_data)))
        if self.predict_data is not None:
            table.add_row("Prediction file path", str(self.predict_set))
            table.add_row("Prediction samples", str(len(self.predict_data)))
        table.add_row("N node attributes", str(self.node_in_channels))
        table.add_row("N edge attributes", str(self.edge_in_channels))
        table.add_row("N glob attributes", str(self.glob_in_channels))
        console.print(table)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          follow_batch=['edge_attr', 'hyperedge_index'] if self._use_hyperedge else ['edge_attr'],
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last,
                          shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data,
                          batch_size=self.batch_size,
                          follow_batch=['edge_attr', 'hyperedge_index'] if self._use_hyperedge else ['edge_attr'],
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last,
                          shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_data,
                          batch_size=self.batch_size,
                          follow_batch=['edge_attr', 'hyperedge_index'] if self._use_hyperedge else ['edge_attr'],
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last,
                          shuffle=False)