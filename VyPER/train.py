import os.path as osp
import hydra
import torch
import lightning.pytorch as pl

from lightning.pytorch.loggers import TensorBoardLogger
from lightning_utilities.core.imports import RequirementCache
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
    DeviceStatsMonitor,
    ModelSummary,
    TQDMProgressBar,
    EarlyStopping
)

from VyPER.data import VyPERDataModule
from VyPER.models import VyPER
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

_RICH_AVAILABLE = RequirementCache("rich>=10.2.2")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def Train(cfg : DictConfig) -> None:
    r"""Perform network training using parameters defined in
    `option_file`.

    Args:
        cfg (str): a `.yaml` file, stores training related parameters. (default: :obj:`str`=None).
    """
    print(OmegaConf.to_yaml(cfg))

    datamodule = VyPERDataModule(
        config = osp.join(hydra.utils.get_original_cwd(), f'configs/{HydraConfig.get().job.config_name}.yaml'),
        train_set = cfg['datasets']['train_set'],
        val_set = cfg['datasets']['val_set'],
        predict_set = None,
        event_filter = cfg['datasets']['event_filter'],
        batch_size = cfg['training']['batch_size'],
        percent_train_samples = cfg['datasets']['train_val_split'],
        drop_last = cfg['datasets']['drop_last'],
        num_workers = cfg['device']['num_workers'],
        pin_memory = True if cfg['device']['accelerator']=="gpu" else False
    )

    # print(datamodule.node_in_channels)

    model = VyPER(
        node_in_channels = len(cfg['input']['node_features'])+1,
        edge_in_channels = len(cfg['input']['edge_features']),
        global_in_channels = len(cfg['input']['global_features']),
        nu_out_channels = len(cfg['target']['neutrinos']['features']),
        message_feats = cfg['network']['message_feats'],
        dropout = cfg['training']['dropout'],
        num_message_layers = cfg['network']['num_message_layers'],
        hyperedge_feats = cfg['network']['hyperedge_feats'],
        hyperedge_order = cfg['network']['hyperedge_order'],
        num_sampling_steps = cfg['network']['num_sampling_steps'],
        num_attn_heads = cfg['network']['num_attn_heads'],
        optimizer = cfg['training']['optimizer'],
        lr = cfg['training']['learning_rate'],
        alpha = cfg['training']['alpha'],
        eta = cfg['training']['eta'],
        reduction = cfg['training']['loss_reduction']
    )

    callbacks = [
        ModelCheckpoint(
            verbose=True,
            monitor="loss/validation_loss",
            save_top_k=1,
            mode="min",
            save_last=True
        ),
        EarlyStopping(
            monitor="loss/validation_loss",
            mode="min",
            min_delta=0.00,
            patience=cfg['training']['patience'],
            verbose=False
        ),
        LearningRateMonitor(),
        DeviceStatsMonitor(),
        RichProgressBar() if _RICH_AVAILABLE else TQDMProgressBar(),
        RichModelSummary(max_depth=2) if _RICH_AVAILABLE else ModelSummary(max_depth=2)
    ]

    trainer = pl.Trainer(
        accelerator = cfg['device']['accelerator'],
        devices = cfg['device']['num_devices'],
        max_epochs = cfg['training']['epochs'],
        callbacks = callbacks,
        gradient_clip_val = cfg['training']['gradient_clip'],
        accumulate_grad_batches = cfg['training']['grad_accum_batches'],
        logger = TensorBoardLogger(save_dir=cfg['training']['save_directory'], name="", log_graph=True)
    )

    if cfg['training']['continue_from_ckpt'] is not None:
        print("Resume training state from %s"%(cfg['training']['continue_from_ckpt']))

    trainer.fit(
        model,
        datamodule = datamodule,
        ckpt_path = cfg['training']['continue_from_ckpt']
    )


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    Train()