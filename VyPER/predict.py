import os
import os.path as osp
import omegaconf
import typing
import numpy
import hydra
import torch
import torch_geometric
import lightning.pytorch as pl
import pandas as pd

from VyPER.data import VyPERDataModule
from VyPER.models import VyPER
from VyPER.io import ckpt_loader, PredictionWriter
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from packaging import version


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def Predict(cfg : DictConfig) -> None:
    r"""Make predictions using the trained model defined in `option_file`.

    Args:
        cfg (str): a `.yaml` file, stores training related parameters. (default: :obj:`str`=None).
    """
    print(OmegaConf.to_yaml(cfg))

    map_location = torch.device('cuda') if cfg['device']['accelerator'].lower() == "gpu" else torch.device('cpu')

    # Load checkpoint
    ckpt_file = ckpt_loader(cfg)

    # Load hyperparameters
    hparams_file = osp.join(cfg['predicting']['model_directory'], "hparams.yaml")
    assert os.path.isfile(hparams_file), f"`hparams.ymal` is not found in {cfg['predicting']['model_directory']}."

    datamodule = VyPERDataModule(
        config = osp.join(hydra.utils.get_original_cwd(), f'configs/{HydraConfig.get().job.config_name}.yaml'),
        predict_set = cfg['datasets']['predict_set'],
        force_reload = cfg['datasets']['force_reload'],
        batch_size = cfg['predicting']['batch_size'],
        num_workers = cfg['device']['num_workers'],
        pin_memory = True if cfg['device']['accelerator']=="gpu" else False
    )

    model = VyPER.load_from_checkpoint(
        checkpoint_path = ckpt_file,
        hparams_file = hparams_file,
        map_location = map_location,
        num_sampling_steps = cfg['predicting']['num_sampling_steps'],
        weights_only = True # Required since PyTorch 2.9.
    )

    writer = PredictionWriter(
        cfg['predicting']['save_as'],
        edge_out_channels=len(cfg['target']['edge'])+1 if 'edge' in cfg['target'].keys() else 0,
        num_neutrinos=int(cfg['target']['topology']['neutrinos']),
        hyperedge_out_channels=len(cfg['target']['hyperedge'])+1 if 'hyperedge' in cfg['target'].keys() else None,
        hyperedge_order=len(list(cfg['target']['hyperedge'].values())[0][0]) if 'hyperedge' in cfg['target'].keys() else None,
        edge_reduction=cfg['predicting']['edge_reduction']
    )

    trainer = pl.Trainer(
        accelerator = cfg['device']['accelerator'],
        devices = cfg['device']['num_devices'],
        callbacks = [writer],
        logger=False
    )

    model.eval()
    with torch.no_grad():
        trainer.predict(model, datamodule=datamodule, return_predictions=False)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    # Required since PyTorch 2.6, see [#53](https://github.com/tzuhanchang/VyPER/pull/53).
    if version.parse(torch.__version__) >= version.parse("2.6"):
        torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr,torch_geometric.data.data.DataTensorAttr,torch_geometric.data.storage.GlobalStorage])
        torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig,omegaconf.base.ContainerMetadata,typing.Any,omegaconf.nodes.AnyNode,omegaconf.base.Metadata,omegaconf.listconfig.ListConfig,int])

    import tqdm
    import multiprocessing
    tqdm.tqdm.monitor_interval = 0
    tqdm.tqdm.set_lock(multiprocessing.RLock())

    Predict()
