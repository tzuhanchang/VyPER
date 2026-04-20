import torch

from torch import optim
from lightning import LightningModule
from torch_geometric.utils import scatter, unbatch, unbatch_edge_index
from torchmetrics.classification import MultilabelAccuracy
from typing import Optional

from .mpnn import MPNNs
from .hyperedge import HyperedgeBlock
from .diffusion import NeutrinoDiffusion
from .loss import EdgeLoss, HyperedgeLoss, DiffusionLoss, CombinedLoss
from VyPER.utils import get_neutrino_p4, unbatch_hyperedge_index


class VyPER(LightningModule):
    def __init__(
        self,
        node_in_channels: int,
        edge_in_channels: int,
        global_in_channels: int,
        edge_out_channels: int,
        hyperedge_out_channels: int,
        nu_out_channels: Optional[int] = None,
        message_feats: int = 32,
        attn_feats: int = 32,
        dropout: float = 0.01,
        num_message_layers: int = 3,
        use_edge: bool = True,
        use_hyperedge: bool = True,
        use_diffusion: bool = True,
        hyperedge_feats: int = 32,
        hyperedge_order: int = 3,
        num_sampling_steps: int = 50,
        num_attn_heads: int = 1,
        num_dit_blocks: int = 4,
        optimizer: str = "Adam",
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        momentum: float = 0.0,
        alpha: float = 0.5,
        eta: float = 0.5,
        reduction: float = 'mean',
        lr_scheduler: Optional[dict] = None,
    ) -> None:

        super().__init__()

        self.save_hyperparameters()

        if self.hparams.use_diffusion:
            assert self.hparams.nu_out_channels is not None

        self.MessagePassing = MPNNs(
            node_in_channels=self.hparams.node_in_channels,
            edge_in_channels=self.hparams.edge_in_channels,
            global_in_channels=self.hparams.global_in_channels,
            node_out_channels=self.hparams.message_feats,
            edge_out_channels=self.hparams.edge_out_channels,
            global_out_channels=self.hparams.message_feats,
            num_layers=self.hparams.num_message_layers,
            message_feats=self.hparams.message_feats,
            nu_ctx_channels=self.hparams.message_feats,
            dropout=self.hparams.dropout,
            use_edge=self.hparams.use_edge,
            use_neutrino=self.hparams.use_diffusion
        )

        if self.hparams.use_diffusion:
            self.NeutrinoDiffusion = NeutrinoDiffusion(
                d_ctx=self.hparams.message_feats,
                d_embed=self.hparams.attn_feats,
                d_target=self.hparams.nu_out_channels,
                num_heads=self.hparams.num_attn_heads,
                num_message_steps=self.hparams.num_message_layers,
                num_dit_blocks=self.hparams.num_dit_blocks,
                num_sampling_steps=self.hparams.num_sampling_steps
            )

        if self.hparams.use_hyperedge:
            self.Hyperedge = HyperedgeBlock(
                node_in_channels=self.hparams.message_feats*self.hparams.num_message_layers,
                node_out_channels=self.hparams.hyperedge_out_channels,
                global_in_channels=self.hparams.message_feats*self.hparams.num_message_layers,
                message_feats=self.hparams.hyperedge_feats,
                dropout=self.hparams.dropout
            )
            self.metric_hyperedge = MultilabelAccuracy(num_labels=self.hparams.hyperedge_out_channels,
                                                       average='none', ignore_index=0)

        if self.hparams.use_edge:
            self.metric_edge = MultilabelAccuracy(num_labels=self.hparams.edge_out_channels,
                                                  average='none', ignore_index=0)

    def forward(self, x, edge_index, edge_attr, u, batch, x_fw_mask=None, edge_fw_mask=None,
                hyperedge_index=None, hyperedge_index_batch=None,
                neutrino_t=None, train_mode=True, sampling=True):
        # Message-passing
        message_out = self.MessagePassing(
            x, edge_index, edge_attr, u, batch, x_fw_mask, edge_fw_mask
        )
        x_out, edge_attr_out, u_out = message_out[0], message_out[1], message_out[2]
        # Neutrino diffusion
        if self.hparams.use_diffusion:
            num_neutrinos = scatter(x_fw_mask, index=batch, dim=0, dim_size=u.size(0), reduce='sum')
            nu_batch = torch.unique(batch).repeat_interleave(num_neutrinos)
            if train_mode:
                nu_out = self.NeutrinoDiffusion(message_out[3], batch, nu_batch, neutrino_t, sampling=sampling)
            else:
                nu_out = self.NeutrinoDiffusion(message_out[3], batch, nu_batch)
        else:
            nu_out, nu_batch = None, None
        # Hyperedge step
        if hyperedge_index is not None and hyperedge_index_batch is not None:
            hyperedge_out, hyperedge_batch = self.Hyperedge(x_out, u_out, batch, hyperedge_index, hyperedge_index_batch, self.hparams.hyperedge_order)
        else:
            hyperedge_out, hyperedge_batch = None, None
        return [edge_attr_out, nu_out, nu_batch, hyperedge_out, hyperedge_batch]

    def configure_optimizers(self):
        if str(self.hparams.optimizer).lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif str(self.hparams.optimizer).lower() == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif str(self.hparams.optimizer).lower() == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum)
        # --------- custom optimizers ---------
        # elif
        # -------------------------------------
        else:
            raise NotImplementedError("Supported optimizers are: `Adam`, `AdamW` and `SGD`.")
        
        if self.hparams.lr_scheduler is not None:
            optimizer = {"optimizer": optimizer,
                         "lr_scheduler": {"scheduler": eval(self.hparams.lr_scheduler['method'])(optimizer, **self.hparams.lr_scheduler['kwargs']),
                                          "interval": "epoch", "monitor": "loss/validation_loss", "frequency": 1, "strict": True}}
        return optimizer

    def training_step(self, train_batch, batch_idx):
        if self.hparams.use_diffusion:
            x_fw_mask, edge_fw_mask = (train_batch.x_fw_mask, train_batch.edge_fw_mask)
        else:
            x_fw_mask, edge_fw_mask = (None, None)

        # Network
        out = self.forward(
            train_batch.x, train_batch.edge_index, train_batch.edge_attr, train_batch.u,
            train_batch.batch, x_fw_mask, edge_fw_mask,
            hyperedge_index=train_batch.hyperedge_index if self.hparams.use_hyperedge else None,
            hyperedge_index_batch=train_batch.hyperedge_index_batch if self.hparams.use_hyperedge else None,
            neutrino_t=train_batch.neutrino_t if self.hparams.use_diffusion else None,
            train_mode=True, sampling=False
        )

        # Compute edge loss
        if self.hparams.use_edge:
            edge_loss = EdgeLoss(out[0], train_batch.edge_attr_t, train_batch.edge_attr_batch,
                                train_batch.topo_max_num_edges)
            self.log('loss/train_edge_loss', edge_loss.mean(), batch_size=len(train_batch),
                    on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        else:
            edge_loss = None

        # Compute diffusion loss
        if self.hparams.use_diffusion:
            nu_loss, nu_batch = out[1], out[2]
            nu_loss = DiffusionLoss(nu_loss, nu_batch, reduction='sum')
            self.log('loss/train_diffusion_loss', nu_loss.mean(), batch_size=len(train_batch),
                 on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        else:
            nu_loss = None

        # Compute hyperedge loss
        if self.hparams.use_hyperedge:
            hyperedge_out, hyperedge_batch = out[3], out[4]
            hyperedge_loss = HyperedgeLoss(hyperedge_out, train_batch.hyperedge_attr_t, hyperedge_batch,
                                           train_batch.topo_max_num_hyperedges)
            self.log('loss/train_hyperedge_loss', hyperedge_loss.mean(), batch_size=len(train_batch),
                 on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        else:
            hyperedge_loss = None

        # Compute total network loss
        loss = CombinedLoss(edge_loss, nu_loss, hyperedge_loss, reduction=self.hparams.reduction,
                            alpha=self.hparams.alpha, eta=self.hparams.eta)
        self.log('loss/train_loss', loss, batch_size=len(train_batch), on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        if self.hparams.use_diffusion:
            x_fw_mask, edge_fw_mask = (val_batch.x_fw_mask, val_batch.edge_fw_mask)
        else:
            x_fw_mask, edge_fw_mask = (None, None)
        final_val_batch_idx = len(self.trainer.datamodule.val_dataloader()) - 1

        # Network
        out = self.forward(
            val_batch.x, val_batch.edge_index, val_batch.edge_attr, val_batch.u,
            val_batch.batch, x_fw_mask, edge_fw_mask,
            hyperedge_index=val_batch.hyperedge_index if self.hparams.use_hyperedge else None,
            hyperedge_index_batch=val_batch.hyperedge_index_batch if self.hparams.use_hyperedge else None,
            neutrino_t=val_batch.neutrino_t if self.hparams.use_diffusion else None,
            train_mode=True,
            sampling=True if batch_idx==final_val_batch_idx else False
        )

        # Compute edge loss and accuracy
        if self.hparams.use_edge:
            edge_loss = EdgeLoss(out[0], val_batch.edge_attr_t, val_batch.edge_attr_batch,
                                val_batch.topo_max_num_edges)
            self.log('loss/validation_edge_loss', edge_loss.mean(), batch_size=len(val_batch),
                    on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            edge_accuracy = self.metric_edge(torch.nn.functional.softmax(out[0], dim=1),
                                            val_batch.edge_attr_t.float())
            for i in range(self.hparams.edge_out_channels):
                self.log(f'accuracy/edge_channel_{i}', edge_accuracy[i], batch_size=len(val_batch), on_step=False, on_epoch=True,
                        prog_bar=False, logger=True, sync_dist=True)
        else:
            edge_loss = None

        # Compute diffusion loss
        if self.hparams.use_diffusion:
            nu_out, nu_batch = out[1], out[2]
            if batch_idx == final_val_batch_idx:
                nu_loss, nu_out = nu_out
            else:
                nu_loss = nu_out
            nu_loss = DiffusionLoss(nu_loss, nu_batch, reduction='sum')
            self.log('loss/validation_diffusion_loss', nu_loss.mean(), batch_size=len(val_batch),
                     on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        else:
            nu_loss = None

        # Compute hyperedge loss and accuracy
        if self.hparams.use_hyperedge:
            hyperedge_out, hyperedge_batch = out[3], out[4]
            hyperedge_loss = HyperedgeLoss(hyperedge_out, val_batch.hyperedge_attr_t, hyperedge_batch,
                                        val_batch.topo_max_num_hyperedges)
            self.log('loss/validation_hyperedge_loss', hyperedge_loss.mean(), batch_size=len(val_batch),
                        on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            hyperedge_accuracy = self.metric_hyperedge(torch.nn.functional.softmax(hyperedge_out, dim=1),
                                                        val_batch.hyperedge_attr_t.float())
            for i in range(self.hparams.hyperedge_out_channels):
                self.log(f'accuracy/hyperedge_channel_{i}', hyperedge_accuracy[i], batch_size=len(val_batch), on_step=False,
                            on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        else:
            hyperedge_loss = None

        # Compute total network loss
        loss = CombinedLoss(edge_loss, nu_loss, hyperedge_loss, reduction=self.hparams.reduction,
                            alpha=self.hparams.alpha, eta=self.hparams.eta)
        self.log('loss/validation_loss', loss, batch_size=len(val_batch), on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        # Log histograms @ final validation step
        if batch_idx == final_val_batch_idx and self.hparams.use_diffusion:
            p = get_neutrino_p4(nu_out,
                                self.trainer.datamodule.neutrino_momentum_func,
                                self.trainer.datamodule.neutrino_4vector_func,
                                self.trainer.datamodule.neutrino_4vector_loc,
                                self.trainer.datamodule.nu_reverse_transform_methods)
            p_t = get_neutrino_p4(val_batch.neutrino_t,
                                self.trainer.datamodule.neutrino_momentum_func,
                                self.trainer.datamodule.neutrino_4vector_func,
                                self.trainer.datamodule.neutrino_4vector_loc,
                                self.trainer.datamodule.nu_reverse_transform_methods)
            dPhi = torch.arctan2(torch.sin(p_t.phi-p.phi),torch.cos(p_t.phi-p.phi))
            dEta = p_t.eta - p.eta
            dR   = torch.sqrt(dPhi*dPhi + dEta*dEta)
            tensorboard = self.logger.experiment
            tensorboard.add_histogram('histograms/px', p.px, global_step=self.current_epoch)
            tensorboard.add_histogram('histograms/py', p.py, global_step=self.current_epoch)
            tensorboard.add_histogram('histograms/pz', p.pz, global_step=self.current_epoch)
            tensorboard.add_histogram('histograms/e', p.e, global_step=self.current_epoch)
            tensorboard.add_histogram('histograms/eta', p.eta, global_step=self.current_epoch)
            tensorboard.add_histogram('histograms/phi', p.phi, global_step=self.current_epoch)
            tensorboard.add_histogram('histograms/pt', p.pt, global_step=self.current_epoch)
            tensorboard.add_histogram('histograms/dPhi', dPhi, global_step=self.current_epoch)
            tensorboard.add_histogram('histograms/dEta', dEta, global_step=self.current_epoch)
            tensorboard.add_histogram('histograms/dR', dR, global_step=self.current_epoch)
            self.log('accuracy/dR_mean', dR.mean(), batch_size=len(val_batch), on_step=False, on_epoch=True,
                 prog_bar=False, logger=True, sync_dist=True)

    def predict_step(self, pred_batch, batch_idx, dataloader_idx=0):
        if self.hparams.use_diffusion:
            x_fw_mask, edge_fw_mask = (pred_batch.x_fw_mask, pred_batch.edge_fw_mask)
        else:
            x_fw_mask, edge_fw_mask = (None, None)

        # Network
        out = self.forward(
            pred_batch.x, pred_batch.edge_index, pred_batch.edge_attr, pred_batch.u,
            pred_batch.batch, x_fw_mask, edge_fw_mask,
            hyperedge_index=pred_batch.hyperedge_index if self.hparams.use_hyperedge else None,
            hyperedge_index_batch=pred_batch.hyperedge_index_batch if self.hparams.use_hyperedge else None,
            neutrino_t=None, train_mode=False, sampling=True
        )
        # Unpack edge
        edge_attr_out = torch.nn.functional.softmax(out[0], dim=1)
        edge_out = unbatch(edge_attr_out, pred_batch.edge_attr_batch, dim=0)
        edge_index = unbatch_edge_index(pred_batch.edge_index, pred_batch.batch,
                                        batch_size=self.trainer.datamodule.batch_size)

        # Unpack neutrino
        if self.hparams.use_diffusion:
            nu_out, nu_batch = out[1], out[2]
            for column in range(nu_out.size(1)):
                nu_out[:,column] = self.trainer.datamodule.nu_reverse_transform_methods[column](nu_out[:,column])
            nu_out = unbatch(nu_out, nu_batch, dim=0)
        else:
            nu_out = None

        # Unpack hyperedge
        if self.hparams.use_hyperedge:
            hyperedge_out, hyperedge_batch = out[3], out[4]
            hyperedge_out = torch.nn.functional.softmax(hyperedge_out, dim=1)
            hyperedge_out = unbatch(hyperedge_out, hyperedge_batch.type(torch.int64))
            hyperedge_index = unbatch_hyperedge_index(pred_batch.hyperedge_index, hyperedge_batch.type(torch.int64))
        else:
            hyperedge_out, hyperedge_index = None, None
        return [edge_out, edge_index, nu_out, hyperedge_out, hyperedge_index]
