import os.path as osp
import hydra
import torch
import torch_geometric
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
from packaging import version

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
        force_reload = cfg['datasets']['force_reload'],
        batch_size = cfg['training']['batch_size'],
        percent_train_samples = cfg['datasets']['train_val_split'],
        drop_last = cfg['datasets']['drop_last'],
        num_workers = cfg['device']['num_workers'],
        pin_memory = True if cfg['device']['accelerator']=="gpu" else False
    )

    _use_edge      = 'edge' in cfg['target'].keys()
    _use_hyperedge = 'hyperedge' in cfg['target'].keys()
    _use_diffusion = 'neutrinos' in cfg['target'].keys()

    model = VyPER(
        node_in_channels = len(cfg['input']['node_features'])+1,
        edge_in_channels = len(cfg['input']['edge_features']),
        global_in_channels = len(cfg['input']['global_features']),
        edge_out_channels = len(cfg['target']['edge'])+1 if _use_edge else 0,
        hyperedge_out_channels = len(cfg['target']['hyperedge'])+1 if _use_hyperedge else None,
        nu_out_channels = len(cfg['target']['neutrinos']['features']) if _use_diffusion else None,
        message_feats = cfg['network']['message_feats'],
        attn_feats = cfg['network']['attn_feats'],
        dropout = cfg['training']['dropout'],
        num_message_layers = cfg['network']['num_message_layers'],
        use_edge = _use_edge,
        use_hyperedge = _use_hyperedge,
        use_diffusion = _use_diffusion,
        hyperedge_feats = cfg['network']['hyperedge_feats'] if _use_hyperedge else None,
        hyperedge_order = cfg['network']['hyperedge_order'] if _use_hyperedge else None,
        num_sampling_steps = cfg['training']['num_sampling_steps'],
        num_attn_heads = cfg['network']['num_attn_heads'],
        num_dit_blocks = cfg['network']['num_dit_blocks'],
        noise_distribution = cfg['network']['noise_distribution'],
        optimizer = cfg['training']['optimizer'],
        lr = cfg['training']['learning_rate'],
        weight_decay = cfg['training']['weight_decay'],
        momentum = cfg['training']['momentum'],
        alpha = cfg['training']['alpha'],
        eta = cfg['training']['eta'],
        reduction = cfg['training']['loss_reduction'],
        lr_scheduler=cfg['training']['lr_scheduler']
    )

    callbacks = [
        ModelCheckpoint(
            verbose=True,
            monitor="loss/validation_loss",
            save_top_k=1,
            mode="min",
            save_last=True,
            filename="epoch={epoch}-loss={loss/validation_loss:.3f}",
            auto_insert_metric_name=False,
            enable_version_counter=False
        ),
        EarlyStopping(
            monitor="loss/validation_loss",
            mode="min",
            min_delta=0.00,
            patience=cfg['training']['patience'],
            verbose=False),
        LearningRateMonitor(),
        DeviceStatsMonitor(),
        RichProgressBar() if _RICH_AVAILABLE else TQDMProgressBar(),
        RichModelSummary(max_depth=2) if _RICH_AVAILABLE else ModelSummary(max_depth=2)
    ]

    if cfg['training']['save_ckpts'] is not None:
        for monitor, mode in cfg['training']['save_ckpts'].items():
            safe_monitor = monitor.replace('/', '_')
            callbacks.append(
                ModelCheckpoint(
                    filename="epoch={epoch}-"+safe_monitor+"={"+monitor+":.3f}",
                    monitor=monitor,
                    mode=mode,
                    save_top_k=1,
                    save_last=False,
                    auto_insert_metric_name=False,
                )
            )

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

    # Required since PyTorch 2.6, see [#53](https://github.com/tzuhanchang/VyPER/pull/53).
    if version.parse(torch.__version__) >= version.parse("2.6"):
        torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr,torch_geometric.data.data.DataTensorAttr,torch_geometric.data.storage.GlobalStorage])

    import tqdm
    import multiprocessing
    tqdm.tqdm.monitor_interval = 0
    tqdm.tqdm.set_lock(multiprocessing.RLock())

    Train()
