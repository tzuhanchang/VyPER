import os
import os.path as osp
import hydra
import torch
import lightning.pytorch as pl

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

    # Load checkpoint
    assert model_directory is not None, "No `model_directory` provided. Abort!"
    ckpt_file = [filename for filename in os.listdir(osp.join(model_directory, "checkpoints")) 
                 if filename.startswith("epoch")]
    if len(ckpt_file) == 1:
        ckpt_file = osp.join(model_directory, "checkpoints", ckpt_file[0])
    elif len(ckpt_file) > 1:
        ckpt_file = osp.join(model_directory, "checkpoints", ckpt_file[-1])
        raise UserWarning(f"There are multiple .ckpt files listed in {model_directory}, using the last checkpoint.")
    elif len(ckpt_file) == 0:
        raise RuntimeError(f"No checkpoint files have been found in {model_directory}.")

    # Load hyperparameters
    hparams_file = osp.join(model_directory, "hparams.yaml")
    assert os.path.isfile(hparams_file), f"`hparams.ymal` is not found in {model_directory}."

    datamodule = VyPERDataModule(
        config = osp.join(hydra.utils.get_original_cwd(), f'configs/{HydraConfig.get().job.config_name}.yaml'),
        predict_set = cfg['datasets']['predict_set'],
        batch_size = cfg['predicting']['batch_size'],
        num_workers = cfg['device']['num_workers'],
        pin_memory = True if cfg['device']['accelerator']=="gpu" else False
    )

    model = VyPER.load_from_checkpoint(
        checkpoint_path = ckpt_file,
        hparams_file = hparams_file,
        map_location = map_location
    )

    writer = PredictionWriter(
        cfg['predicting']['save_as'],
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