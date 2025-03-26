import h5py
import torch

import os.path as osp
import numpy as np

from tqdm import tqdm
from lightning.pytorch.callbacks import BasePredictionWriter
from typing import Optional

from VyPER.utils import edge_reduction


class PredictionWriter(BasePredictionWriter):
    r"""Write prediction results.

    Args:
        output_dir (str): Path where the results are dumped.

    :type: :obj:`None`
    """
    def __init__(self, output_dir, edge_reduction: Optional[str]=None) -> None:
        super().__init__(write_interval='batch')

        self.output_dir = output_dir
        self.edge_reduction = edge_reduction
        self.num_pred_events = None
        self.batch_size = None

        if osp.exists(self.output_dir):
            raise ValueError(f"{self.output_dir} exists. Abort!")

        self.output_file = h5py.File(self.output_dir, mode='w')

    def prepare_output_file(self) -> None:
        raw_group = self.output_file.create_group('VyPER')

        index_dtype = h5py.vlen_dtype(np.dtype('int64'))
        value_dtype = h5py.vlen_dtype(np.dtype('float32'))

        self.edge_index = raw_group.create_dataset(
            "EdgeIndex", (self.num_pred_events,2,),dtype=index_dtype)
        self.edge_out = raw_group.create_dataset(
            "EdgeSoftP", (self.num_pred_events,1,), dtype=value_dtype)
            # TODO: This need to be updated for multiclass 
        self.neutrino_out = raw_group.create_dataset(
            "Neutrino", (self.num_pred_events,2,), dtype=value_dtype)
            # TODO: This need to be updated for different neutrino count

    @torch.no_grad()
    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, 
                           batch, batch_idx, dataloader_idx) -> None:
        # Prepare the output file if it is not initiated
        if self.num_pred_events is None:
            self.batch_size = trainer.datamodule.batch_size
            self.num_pred_events = len(trainer.datamodule.predict_data)
            self.prepare_output_file()

        edge_out, edge_index, nu_out = prediction

        _num_events = len(edge_out)
        for i in tqdm(range(_num_events),
                      desc=f"Evaluating batch {batch_idx}", unit='event',leave=False):
            idx_save = batch_indices[i]

            # Edge reduction
            if self.edge_reduction is not None:
                reduced_edge, reduced_edge_index = edge_reduction(
                    edge_out[i], edge_index[i], reduction=self.edge_reduction,
                    return_reduced_indices=True)
            else:
                reduced_edge = edge_out[i]
                reduced_edge_index = edge_index[i]

            # Write output
            self.edge_out[idx_save] = reduced_edge.transpose(0,1).detach().cpu().numpy()
            self.edge_index[idx_save] = reduced_edge_index.detach().cpu().numpy()
            self.neutrino_out[idx_save] = nu_out[i].detach().cpu().numpy()

        # Close the file when the final event is written
        if (idx_save + 1) == self.num_pred_events:
            self.output_file.close()