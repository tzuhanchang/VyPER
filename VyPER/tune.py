import os.path as osp
import hydra
import optuna
import torch
import lightning.pytorch as pl

from lightning.pytorch.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback
from VyPER.data import VyPERDataModule
from VyPER.models import VyPER
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

CONFIG_PATH = None
CONFIGS = None


def objective(trial: optuna.trial.Trial) -> float:
    r"""An `objective` function to be optimised.

    Args:
        trial (optuna.trial.Trial): is used to suggest hyperparameter values.
        option_file (str, optional): `.json` file, stores training related
            parameters. (default: :obj:`str`=None)
    """
    for key, value in CONFIGS['tune'].items():
        if key not in CONFIGS.keys():
            continue
        for parm, tune_settings in value.items():
            var_type = str(tune_settings[0]).lower()
            lower, upper = tune_settings[1], tune_settings[2]
            try:
                step = tune_settings[3]
                _with_step = True
            except IndexError:
                step = None
                _with_step = False

            if var_type == 'int':
                if _with_step:
                    CONFIGS[key][parm] = trial.suggest_int(parm, lower, upper, step=step)
                else:
                    CONFIGS[key][parm] = trial.suggest_int(parm, lower, upper)
            elif var_type == 'float':
                CONFIGS[key][parm] = trial.suggest_float(parm, lower, upper, step=step)
            elif var_type == 'loguniform':
                CONFIGS[key][parm] = trial.suggest_float(parm, lower, upper, log=True)
            else:
                raise NotImplementedError('Hyperparameter dtype not supported.')

    datamodule = VyPERDataModule(
        config = CONFIG_PATH,
        train_set = CONFIGS['datasets']['tune_set'],
        val_set = None,
        predict_set = None,
        event_filter = CONFIGS['datasets']['event_filter'],
        cache_dir = CONFIGS['datasets']['cache_dir'],
        force_reload = False,
        batch_size = CONFIGS['training']['batch_size'],
        percent_train_samples = CONFIGS['datasets']['train_val_split'],
        drop_last = CONFIGS['datasets']['drop_last'],
        num_workers = CONFIGS['device']['num_workers'],
        pin_memory = True if CONFIGS['device']['accelerator']=="gpu" else False
    )

    _use_hyperedge = 'hyperedge' in CONFIGS['target'].keys()

    try:
        model = VyPER(
            node_in_channels = len(CONFIGS['input']['node_features'])+1,
            edge_in_channels = len(CONFIGS['input']['edge_features']),
            global_in_channels = len(CONFIGS['input']['global_features']),
            edge_out_channels = len(CONFIGS['target']['edge'])+1,
            hyperedge_out_channels = len(CONFIGS['target']['hyperedge'])+1 if _use_hyperedge else None,
            nu_out_channels = len(CONFIGS['target']['neutrinos']['features']),
            message_feats = CONFIGS['network']['message_feats'],
            dropout = CONFIGS['training']['dropout'],
            num_message_layers = CONFIGS['network']['num_message_layers'],
            use_hyperedge = _use_hyperedge,
            hyperedge_feats = CONFIGS['network']['hyperedge_feats'] if _use_hyperedge else None,
            hyperedge_order = CONFIGS['network']['hyperedge_order'] if _use_hyperedge else None,
            num_sampling_steps = CONFIGS['network']['num_sampling_steps'],
            num_attn_heads = CONFIGS['network']['num_attn_heads'],
            optimizer = CONFIGS['training']['optimizer'],
            lr = CONFIGS['training']['learning_rate'],
            weight_decay = CONFIGS['training']['weight_decay'],
            alpha = CONFIGS['training']['alpha'],
            eta = CONFIGS['training']['eta'],
            reduction = CONFIGS['training']['loss_reduction']
        )

        trainer = pl.Trainer(
            accelerator = CONFIGS['device']['accelerator'],
            devices = CONFIGS['device']['num_devices'],
            max_epochs = CONFIGS['tune']['tuner_settings']['epochs'],
            enable_checkpointing=False,
            logger = TensorBoardLogger(
                save_dir=CONFIGS['tune']['tuner_settings']['save_dir'],
                name="",
                log_graph=True
            ),
            callbacks=[
                PyTorchLightningPruningCallback(
                    trial,
                    monitor=CONFIGS['tune']['tuner_settings']['monitor'])
                ]
        )

        trainer.fit(model, datamodule=datamodule)
    except Exception as e:
        print(f"Trial {trial.number} failed: {str(e)}")
        trial.set_user_attr("failed", True)
        trial.set_user_attr("error_message", str(e))
        raise optuna.TrialPruned()

    return trainer.callback_metrics[CONFIGS['tune']['tuner_settings']['monitor']].item()


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def Tune(cfg : DictConfig) -> None:
    global CONFIG_PATH
    global CONFIGS

    CONFIG_PATH = osp.join(hydra.utils.get_original_cwd(), f'configs/{HydraConfig.get().job.config_name}.yaml')
    CONFIGS = cfg

    study = optuna.create_study(
        storage        = CONFIGS['tune']['tuner_settings']['sqlite'],
        study_name     = CONFIGS['tune']['tuner_settings']['study_name'],
        direction      = CONFIGS['tune']['tuner_settings']['direction'],
        load_if_exists = True,
        pruner         = optuna.pruners.MedianPruner()
    )

    study.optimize(objective, n_trials=CONFIGS['tune']['tuner_settings']['n_trials'])

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    Tune()
