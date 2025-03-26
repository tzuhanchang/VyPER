import torch

from torch import nn, optim
from lightning import LightningModule
from torch_geometric.utils import scatter, unbatch, unbatch_edge_index
from torchmetrics.classification import BinaryAccuracy
from typing import Optional

from .mpnn import MPNNs
from .diffusion import NeutrinoDiffusion
from .loss import EdgeLoss, HyperedgeLoss, DiffusionLoss, CombinedLoss
from VyPER.utils import get_neutrino_p4


class VyPER(LightningModule):
    def __init__(
        self,
        node_in_channels: int,
        edge_in_channels: int,
        global_in_channels: int,
        nu_out_channels: int,
        message_feats: int = 32,
        dropout: float = 0.01,
        num_message_layers: int = 3,
        hyperedge_feats: int = 32,
        hyperedge_order: int = 3,
        num_sampling_steps: int = 50,
        num_attn_heads: int = 1,
        optimizer: str = "Adam",
        lr: float = 1e-3,
        alpha: float = 0.5,
        reduction: float = 'mean'
    ) -> None:

        super().__init__()

        self.save_hyperparameters()

        self.MessagePassing = MPNNs(
            node_in_channels=self.hparams.node_in_channels,
            edge_in_channels=self.hparams.edge_in_channels,
            global_in_channels=self.hparams.global_in_channels,
            node_out_channels=self.hparams.message_feats,
            edge_out_channels=1,
            global_out_channels=self.hparams.message_feats,
            num_layers=self.hparams.num_message_layers,
            message_feats=self.hparams.message_feats,
            nu_ctx_channels=self.hparams.message_feats,
            dropout=self.hparams.dropout
        )

        self.NeutrinoDiffusion = NeutrinoDiffusion(
            d_ctx=self.hparams.message_feats,
            d_embed=self.hparams.message_feats,
            d_target=self.hparams.nu_out_channels,
            num_heads=self.hparams.num_attn_heads,
            num_message_steps=self.hparams.num_message_layers,
            num_sampling_steps=self.hparams.num_sampling_steps
        )

        self.metric_edge = BinaryAccuracy(ignore_index=0)

    def forward(self, x, edge_index, edge_attr, u, batch, x_fw_mask, edge_fw_mask,
                neutrino_t=None, train_mode=True, sampling=True):
        # Message-passing
        x_out, edge_attr_out, u_out, nu_ctx = self.MessagePassing(
            x, edge_index, edge_attr, u, batch, x_fw_mask, edge_fw_mask
        )
        edge_attr_out = nn.functional.sigmoid(edge_attr_out)
        # Neutrino diffusion
        num_neutrinos = scatter(x_fw_mask, index=batch, dim=0, dim_size=u.size(0), reduce='sum')
        nu_batch = torch.unique(batch).repeat_interleave(num_neutrinos)
        if train_mode:
            nu_out = self.NeutrinoDiffusion(nu_ctx, batch, nu_batch, neutrino_t, sampling=sampling)
        else:
            nu_out = self.NeutrinoDiffusion(nu_ctx, batch, nu_batch)

        # Hyperedge Finding Step - Currently disabled
        # x_hat, batch_hyperedge  = self.Hyperedge(x_prime, u_prime, batch, edge_index_h, edge_index_h_batch, self.hparams.hyperedge_order)
        return edge_attr_out, nu_out, nu_batch

    def configure_optimizers(self):
        if str(self.hparams.optimizer).lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        # --------- custom optimizers ---------
        # elif
        # -------------------------------------
        else:
            raise NotImplementedError("Supported optimizers are: `torch.Adam`.")
        return optimizer

    def training_step(self, train_batch, batch_idx):
        edge_attr_out, nu_loss, nu_batch = self.forward(
            train_batch.x, train_batch.edge_index, train_batch.edge_attr, train_batch.u,
            train_batch.batch, train_batch.x_fw_mask, train_batch.edge_fw_mask,
            train_batch.neutrino_t, train_mode=True, sampling=False)
        
        edge_loss = EdgeLoss(edge_attr_out, train_batch.edge_attr_t, train_batch.edge_attr_batch)
        nu_loss = DiffusionLoss(nu_loss, nu_batch, reduction='sum')
        loss = CombinedLoss(edge_loss, nu_loss, reduction=self.hparams.reduction)

        # Logging
        self.log('loss/train_edge_loss', edge_loss.mean(), batch_size=len(train_batch),
                 on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('loss/train_diffusion_loss', nu_loss.mean(), batch_size=len(train_batch),
                 on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('loss/train_loss', loss, batch_size=len(train_batch), on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        final_val_batch_idx = len(self.trainer.datamodule.val_dataloader()) - 1
        if batch_idx == final_val_batch_idx:
            edge_attr_out, nu_out, nu_batch = self.forward(
                val_batch.x, val_batch.edge_index, val_batch.edge_attr, val_batch.u,
                val_batch.batch, val_batch.x_fw_mask, val_batch.edge_fw_mask,
                val_batch.neutrino_t, train_mode=True, sampling=True)
            nu_loss, nu_out = nu_out
        else:
            edge_attr_out, nu_loss, nu_batch = self.forward(
                val_batch.x, val_batch.edge_index, val_batch.edge_attr, val_batch.u,
                val_batch.batch, val_batch.x_fw_mask, val_batch.edge_fw_mask,
                val_batch.neutrino_t, train_mode=True, sampling=False)

        edge_loss = EdgeLoss(edge_attr_out, val_batch.edge_attr_t, val_batch.edge_attr_batch)
        nu_loss = DiffusionLoss(nu_loss, nu_batch, reduction='sum')
        loss = CombinedLoss(edge_loss, nu_loss, reduction=self.hparams.reduction)

        edge_accuracy = self.metric_edge(edge_attr_out.flatten(), val_batch.edge_attr_t.float().flatten())

        # Logging
        self.log('loss/validation_edge_loss', edge_loss.mean(), batch_size=len(val_batch),
                 on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('loss/validation_diffusion_loss', nu_loss.mean(), batch_size=len(val_batch),
                 on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('loss/validation_loss', loss, batch_size=len(val_batch), on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log('accuracy/edge', edge_accuracy, batch_size=len(val_batch), on_step=False, on_epoch=True,
                 prog_bar=False, logger=True, sync_dist=True)

        if batch_idx == final_val_batch_idx:
            p = get_neutrino_p4(nu_out, self.trainer.datamodule.nu_reverse_transform_methods)
            tensorboard = self.logger.experiment
            tensorboard.add_histogram('histograms/px', p.px, global_step=self.current_epoch)
            tensorboard.add_histogram('histograms/py', p.py, global_step=self.current_epoch)
            tensorboard.add_histogram('histograms/pz', p.pz, global_step=self.current_epoch)
            tensorboard.add_histogram('histograms/e', p.e, global_step=self.current_epoch)
            tensorboard.add_histogram('histograms/eta', p.eta, global_step=self.current_epoch)
            tensorboard.add_histogram('histograms/phi', p.phi, global_step=self.current_epoch)
            tensorboard.add_histogram('histograms/pt', p.pt, global_step=self.current_epoch)

    def predict_step(self, pred_batch, batch_idx, dataloader_idx=0):
        edge_attr_out, nu_out, nu_batch = self.forward(
            pred_batch.x, pred_batch.edge_index, pred_batch.edge_attr, pred_batch.u,
            pred_batch.batch, pred_batch.x_fw_mask, pred_batch.edge_fw_mask,
            neutrino_t=None, train_mode=False, sampling=True)

        for column in range(nu_out.size(1)):
            nu_out[:,column] = self.trainer.datamodule.nu_reverse_transform_methods[column](nu_out[:,column])

        edge_out = unbatch(edge_attr_out, pred_batch.edge_attr_batch, dim=0)
        edge_index = unbatch_edge_index(pred_batch.edge_index, pred_batch.batch,
                                        batch_size=self.trainer.datamodule.batch_size)
        nu_out = unbatch(nu_out, nu_batch, dim=0)
        return edge_out, edge_index, nu_out