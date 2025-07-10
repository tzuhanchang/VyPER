import os
import os.path as osp
import numpy
import hydra
import torch
import lightning.pytorch as pl
import pandas as pd

from VyPER.data import VyPERDataModule
from VyPER.models import VyPER
from VyPER.io import PredictionWriter
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def Predict(cfg : DictConfig) -> None:
    r"""Make predictions using the trained model defined in `option_file`.

    Args:
        cfg (str): a `.yaml` file, stores training related parameters. (default: :obj:`str`=None).
    """
    print(OmegaConf.to_yaml(cfg))

    map_location = torch.device('cuda') if cfg['device']['accelerator'].lower() == "gpu" else torch.device('cpu')
    model_directory = cfg['predicting']['model_directory']
    model_choice = cfg['predicting']['model_choice'].split('-')

    # Load checkpoint
    if len(model_choice) != 2:
        raise UserWarning(f"Invalid `model_choice`: {cfg['predicting']['model_choice']}, use the last checkpoint instead.")
        ckpt_file = osp.join(model_directory, 'checkpoints', 'last.ckpt')
    else:
        assert model_directory is not None, "No `model_directory` provided. Abort!"
        ckpt_files = [filename.strip('.ckpt').split('-') for filename in
                    os.listdir(osp.join(model_directory, "checkpoints")) if filename.startswith("epoch")]
        ckpt_db = pd.DataFrame([{k: float(v) for k, v in (item.split('=') for item in entry)} for entry in ckpt_files])

        ckpt_idx = getattr(numpy, 'arg'+model_choice[0])(ckpt_db[model_choice[1]])
        print(f"Loading checkpoint: {'-'.join(ckpt_files[ckpt_idx])+'.ckpt'}.")
        ckpt_file = osp.join(model_directory, 'checkpoints', '-'.join(ckpt_files[ckpt_idx])+'.ckpt')

    # Load hyperparameters
    hparams_file = osp.join(model_directory, "hparams.yaml")
    assert os.path.isfile(hparams_file), f"`hparams.ymal` is not found in {model_directory}."

    datamodule = VyPERDataModule(
        config = osp.join(hydra.utils.get_original_cwd(), f'configs/{HydraConfig.get().job.config_name}.yaml'),
        predict_set = cfg['datasets']['predict_set'],
        cache_dir = cfg['datasets']['cache_dir'],
        force_reload = cfg['datasets']['force_reload'],
        batch_size = cfg['predicting']['batch_size'],
        num_workers = cfg['device']['num_workers'],
        pin_memory = True if cfg['device']['accelerator']=="gpu" else False
    )

    model = VyPER.load_from_checkpoint(
        checkpoint_path = ckpt_file,
        hparams_file = hparams_file,
        map_location = map_location,
        num_sampling_steps = cfg['predicting']['num_sampling_steps']
    )

    writer = PredictionWriter(
        cfg['predicting']['save_as'],
        edge_out_channels=len(cfg['target']['edge'])+1,
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
    Predict()
