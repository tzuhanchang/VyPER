import os
import os.path as osp
import hydra
import torch
import omegaconf
import typing
import torch_geometric
import onnxruntime

from omegaconf import DictConfig, OmegaConf
from lightning_utilities.core.imports import RequirementCache
from hydra.core.hydra_config import HydraConfig
from packaging import version

from VyPER.io import ckpt_loader
from VyPER.data import VyPERDataModule
from VyPER.models import VyPER

_ONNX_AVAILABLE = RequirementCache("onnx")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def ONNX(cfg : DictConfig) -> None:
    r"""Perform VyPER model onnx export.
    `option_file`.

    Args:
        cfg (str): a `.yaml` file, stores training related parameters. (default: :obj:`str`=None).
    """
    print(OmegaConf.to_yaml(cfg))

    # Always use CPU for ONNX export
    map_location = torch.device('cpu')

    # Load checkpoint
    ckpt_file = ckpt_loader(cfg, key='onnx_export')

    # Load hyperparameters
    hparams_file = osp.join(cfg['onnx_export']['model_directory'], "hparams.yaml")
    assert os.path.isfile(hparams_file), f"`hparams.ymal` is not found in {cfg['predicting']['model_directory']}."

    assert cfg['datasets']['predict_set'] is not None, "Please provide an example dataset as `datasets.predict_set`."

    datamodule = VyPERDataModule(
        config = osp.join(hydra.utils.get_original_cwd(), f'configs/{HydraConfig.get().job.config_name}.yaml'),
        predict_set = cfg['datasets']['predict_set'],
        batch_size = 2, num_workers = 0, pin_memory = False)
    
    input_samples  = {'x': None, 'edge_index': None, 'edge_attr': None, 'u': None, 'batch': None}
    dynamic_shapes = {'x': {0: "n_nodes"}, 'edge_index': {1: "n_edges"}, 'edge_attr': {0: "n_edges"},
                      'u': {0: "n_graphs"}, 'batch': {0: "n_nodes"}}
    output_names   = []
    # If edge module is used
    if 'edge' in cfg['target'].keys():
        output_names.append('edge_attr_out')
    # If diffusion module is used
    if 'neutrinos' in cfg['target'].keys():
        input_samples.update({'x_fw_mask': None, 'edge_fw_mask': None})
        dynamic_shapes.update({'x_fw_mask': {0: "n_nodes"}, 'edge_fw_mask': {0: "n_edges"}})
        output_names.append('nu_out')
        output_names.append('nu_batch')
    # If hyperedge module is used
    if 'hyperedge' in cfg['target'].keys():
        input_samples.update({'hyperedge_index': None, 'hyperedge_index_batch': None})
        dynamic_shapes.update({'hyperedge_index': {1: "n_hyperedges"}, 'hyperedge_index_batch': {0: "n_hyperedges"}})
        output_names.append('hyperedge_out')
        output_names.append('hyperedge_batch')

    datamodule.setup("fit")
    for data in datamodule.predict_dataloader():
        for key in input_samples.keys():
            input_samples[key] = getattr(data, key)
        break   # terminate after one batch

    print(f"Exporting model '{ckpt_file}' to ONNX.")

    # Load model
    model = VyPER.load_from_checkpoint(
        checkpoint_path = ckpt_file,
        hparams_file = hparams_file,
        map_location = map_location,
        num_sampling_steps = cfg['predicting']['num_sampling_steps'],
        weights_only = False # Required since PyTorch 2.9.
    )

    model = model.eval()
    with torch.no_grad():
        torch.onnx.export(model, tuple(input_samples.values()),
                          input_names=list(input_samples.keys()),
                          output_names=output_names,
                          f=cfg['onnx_export']['save_as'],
                          dynamic_shapes=dynamic_shapes,
                          dynamo=True,
                          opset_version=18,
                          export_params=True)

    print(f"ONNX export successful! ONNX model has been saved to '{cfg['onnx_export']['save_as']}'.")

    if cfg['onnx_export']['consistence_check']['run_check']:
        rtol = cfg['onnx_export']['consistence_check']['rtol']
        atol = cfg['onnx_export']['consistence_check']['atol']
        print(f"Running consistence check, comparing `pytorch` outputs with `onnxruntime` outputs with an relative tolerance of {rtol} and an absolute tolerance of {atol}.")
        pt = model.forward(**input_samples)

        ort_session = onnxruntime.InferenceSession(cfg['onnx_export']['save_as'], providers=["CPUExecutionProvider"])
        onnx_inputs = {key: value.numpy(force=True) for key, value in input_samples.items()}
        ort = ort_session.run(None, onnx_inputs)

        for idx, tensor in enumerate(pt):
            if tensor is not None:
                if idx == 1:    # Fixed neturino output loc in forward function
                    print("Skipping consistence check for neutrino output due to the randomness of diffusion.")
                    continue
                assert len(tensor) == len(ort[idx])
                torch.testing.assert_close(tensor, torch.tensor(ort[idx]), rtol=rtol, atol=atol,
                                           msg=f"Consistence check failed on output {idx} with the given tolerances.")

        print(f"ONNX consistence check successful!")


if __name__ == "__main__":
    # Required since PyTorch 2.6, see [#53](https://github.com/tzuhanchang/VyPER/pull/53).
    if version.parse(torch.__version__) >= version.parse("2.6"):
        torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr,torch_geometric.data.data.DataTensorAttr,torch_geometric.data.storage.GlobalStorage])
        torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig,omegaconf.base.ContainerMetadata,typing.Any,omegaconf.nodes.AnyNode,omegaconf.base.Metadata,omegaconf.listconfig.ListConfig,int])

    if not _ONNX_AVAILABLE:
        raise ModuleNotFoundError(f"Requires `onnx` to be installed.")

    ONNX()