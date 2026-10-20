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
        edge_out_channels (int, optional): number of edge classes, :obj:`None` if
            edges are not predicted. (default: :obj:`None`)
        nu_out_channels (int, optional): number of neutrino features, :obj:`None` if
            neutrinos are not predicted. The `Neutrino` dataset has a shape of
            [num_events, `nu_out_channels`], each entry holding one value per neutrino
            in the event. (default: :obj:`None`)
        hyperedge_out_channels (int, optional): number of hyperedge classes, :obj:`None`
            if hyperedges are not predicted. (default: :obj:`None`)
        hyperedge_order (int, optional): hyperedge order, required if
            :obj:`hyperedge_out_channels` is given. (default: :obj:`None`)
        edge_reduction (str, optional): edge reduction method. (default: :obj:`None`)

    :type: :obj:`None`
    """
    def __init__(
            self,
            output_dir,
            global_out_channels: Optional[int]=None,
            edge_out_channels: Optional[int]=None,
            nu_out_channels: Optional[int]=None,
            hyperedge_out_channels: Optional[int]=None,
            hyperedge_order: Optional[int]=None,
            edge_reduction: Optional[str]=None
        ) -> None:
        super().__init__(write_interval='batch')

        self.output_dir = output_dir
        self.global_out_channels = global_out_channels
        self.edge_out_channels = edge_out_channels
        self.nu_out_channels = nu_out_channels
        self.hyperedge_out_channels = hyperedge_out_channels
        self.hyperedge_order = hyperedge_order
        self.edge_reduction = edge_reduction
        self.num_pred_events = None
        self.batch_size = None

        if osp.exists(self.output_dir):
            raise ValueError(f"{self.output_dir} exists. Abort!")

        self.output_file = h5py.File(self.output_dir, mode='w')

    def prepare_output_file(self) -> None:
        data_group = self.output_file.create_group('VyPER')

        index_dtype = h5py.vlen_dtype(np.dtype('int64'))
        value_dtype = h5py.vlen_dtype(np.dtype('float32'))

        if self.global_out_channels is not None:
            self.classification_out = data_group.create_dataset(
                "Classification", (self.num_pred_events,self.global_out_channels),dtype=np.dtype('float32'))
        if self.edge_out_channels is not None:
            self.edge_index = data_group.create_dataset(
                "EdgeIndex", (self.num_pred_events,2),dtype=index_dtype)
            self.edge_out = data_group.create_dataset(
                "EdgeSoftP", (self.num_pred_events,self.edge_out_channels), dtype=value_dtype)
        if self.nu_out_channels is not None:
            self.neutrino_out = data_group.create_dataset(
                "Neutrino", (self.num_pred_events,self.nu_out_channels), dtype=value_dtype)
        if self.hyperedge_out_channels is not None:
            assert self.hyperedge_order is not None
            self.hyperedge_index = data_group.create_dataset(
                "HyperedgeIndex", (self.num_pred_events,self.hyperedge_order), dtype=index_dtype)
            self.hyperedge_out = data_group.create_dataset(
                "HyperedgeSoftP", (self.num_pred_events,self.hyperedge_out_channels), dtype=value_dtype)

    @torch.no_grad()
    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, 
                           batch, batch_idx, dataloader_idx) -> None:
        r"""Write the prediction results to the opened HDF5 file.

        Note:
            Network output is an ordered list, has the following format:
            [edge_out, edge_index, nu_out, hyperedge_out, hyperedge_index]
        """
        # Prepare the output file if it is not initiated
        if self.num_pred_events is None:
            self.batch_size = trainer.datamodule.batch_size
            self.num_pred_events = len(trainer.datamodule.predict_data)
            self.prepare_output_file()

        if self.edge_out_channels is not None:
            edge_out, edge_index = prediction[0], prediction[1]
        if self.nu_out_channels is not None:
            nu_out = prediction[2]
        if self.hyperedge_out_channels is not None:
            hyperedge_out, hyperedge_index = prediction[3], prediction[4]
        if self.global_out_channels is not None:
            u_out = prediction[5]

        # Event loop
        for i in tqdm(range(len(batch)),
                      desc=f"Evaluating batch {batch_idx}", unit='event',leave=False):
            idx_save = batch_indices[i]

            # Edge reduction
            if self.edge_out_channels is not None:
                if self.edge_reduction is not None:
                    reduced_edge, reduced_edge_index = edge_reduction(
                        edge_out[i], edge_index[i], reduction=self.edge_reduction,
                        return_reduced_indices=True)
                else:
                    reduced_edge = edge_out[i]
                    reduced_edge_index = edge_index[i]
                self.edge_out[idx_save] = reduced_edge.transpose(0,1).detach().cpu().numpy()
                self.edge_index[idx_save] = reduced_edge_index.detach().cpu().numpy()

            if self.nu_out_channels is not None:
                self.neutrino_out[idx_save] = nu_out[i].transpose(0,1).detach().cpu().numpy()

            if self.hyperedge_out_channels is not None:
                self.hyperedge_out[idx_save] = hyperedge_out[i].transpose(0,1).detach().cpu().numpy()
                self.hyperedge_index[idx_save] = hyperedge_index[i].detach().cpu().numpy()

            if self.global_out_channels is not None:
                self.classification_out[idx_save] = u_out[i].detach().cpu().numpy()

        # Close the file when the final event is written
        if (idx_save + 1) == self.num_pred_events:
            self.output_file.close()