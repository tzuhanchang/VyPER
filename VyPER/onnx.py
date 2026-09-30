import os
import os.path as osp
import time
import logging
import warnings
import typing
import subprocess

import hydra
import numpy as np
import omegaconf
import torch
import torch_geometric

from omegaconf import DictConfig, OmegaConf
from lightning_utilities.core.imports import RequirementCache
from hydra.core.hydra_config import HydraConfig
from packaging import version
from pathlib import Path
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from VyPER.io import ckpt_loader
from VyPER.data import VyPERDataModule
from VyPER.models import VyPER

_REQUIREMENTS = {name: RequirementCache(name) for name in ("onnx", "onnxscript", "onnxruntime")}


class VyPERONNX(torch.nn.Module):
    r"""A wrapper of a trained VyPER model, which defines the inputs and
    outputs of the exported ONNX model. The inputs are those of the VyPER
    model, plus `num_sampling_steps` and `noise` (optional). The outputs are
    `edge_attr_out` (logits), `nu_out` (before `reverse_transforms`), `nu_batch`
    and `hyperedge_out` (logits). Only those of the enabled modules are included.

    Args:
        model (VyPER): the trained model.

    :rtype: :class:`Tuple[Tensor,...]`
    """
    def __init__(self, model: VyPER) -> None:
        super().__init__()
        self.model = model
        self.use_edge = bool(model.hparams.use_edge)
        self.use_diffusion = bool(model.hparams.use_diffusion)
        self.use_hyperedge = bool(model.hparams.use_hyperedge)

    @property
    def output_names(self) -> List[str]:
        return (['edge_attr_out'] if self.use_edge else []) \
            + (['nu_out', 'nu_batch'] if self.use_diffusion else []) \
            + (['hyperedge_out'] if self.use_hyperedge else [])

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, u: Tensor, batch: Tensor,
                x_fw_mask: Optional[Tensor] = None, edge_fw_mask: Optional[Tensor] = None,
                hyperedge_index: Optional[Tensor] = None, hyperedge_index_batch: Optional[Tensor] = None,
                num_sampling_steps: Optional[Tensor] = None, noise: Optional[Tensor] = None) -> Tuple[Tensor, ...]:
        edge_attr_out, nu_out, nu_batch, hyperedge_out, _ = self.model(
            x, edge_index, edge_attr, u, batch, x_fw_mask, edge_fw_mask,
            hyperedge_index=hyperedge_index, hyperedge_index_batch=hyperedge_index_batch,
            train_mode=False, noise=noise, num_sampling_steps=num_sampling_steps)
        return tuple(t for t, used in ((edge_attr_out, self.use_edge), (nu_out, self.use_diffusion),
                                       (nu_batch, self.use_diffusion), (hyperedge_out, self.use_hyperedge)) if used)

    def example_inputs(self, data, num_sampling_steps: int, noise_as_input: bool) -> Dict[str, Tensor]:
        r"""Build the model inputs from a batch of the `VyPERDataModule`.

        Args:
            data (torch_geometric.data.Batch): a batch of events.
            num_sampling_steps (int): number of sampling steps.
            noise_as_input (bool): add the initial noise to the inputs.

        :rtype: :class:`Dict[str,Tensor]`
        """
        keys = ['x', 'edge_index', 'edge_attr', 'u', 'batch']
        keys += ['x_fw_mask', 'edge_fw_mask'] if self.use_diffusion else []
        keys += ['hyperedge_index', 'hyperedge_index_batch'] if self.use_hyperedge else []
        inputs = {key: getattr(data, key) for key in keys}
        if self.use_diffusion:
            inputs['num_sampling_steps'] = torch.tensor(num_sampling_steps, dtype=torch.int64)
            if noise_as_input:
                n_nu = int(data.x_fw_mask.to(torch.bool).sum())
                inputs['noise'] = self.model.NeutrinoDiffusion.sample_noise(n_nu)
        return inputs

    @staticmethod
    def dynamic_shapes(inputs: Dict[str, Tensor]) -> Dict[str, Optional[Dict[int, str]]]:
        r"""Dynamic dimensions of :obj:`inputs`.

        Args:
            inputs (Dict[str,Tensor]): model inputs.

        :rtype: :class:`Dict[str,Optional[Dict[int,str]]]`
        """
        shapes = {'x': {0: "n_nodes"}, 'edge_index': {1: "n_edges"}, 'edge_attr': {0: "n_edges"},
                  'u': {0: "n_graphs"}, 'batch': {0: "n_nodes"},
                  'x_fw_mask': {0: "n_nodes"}, 'edge_fw_mask': {0: "n_edges"},
                  'hyperedge_index': {1: "n_hyperedges"}, 'hyperedge_index_batch': {0: "n_hyperedges"},
                  'num_sampling_steps': None, 'noise': {0: "n_neutrinos"}}
        return {key: shapes[key] for key in inputs}


def export_onnx(wrapper: VyPERONNX, inputs: Dict[str, Tensor], save_as: str, opset_version: int,
                metadata: Optional[Dict[str, str]] = None) -> None:
    r"""Export a VyPER model to a single `.onnx` file.

    Args:
        wrapper (VyPERONNX): the model to export.
        inputs (Dict[str,Tensor]): example inputs.
        save_as (str): output file.
        opset_version (int): ONNX opset version.
        metadata (Dict[str,str], optional): model metadata. (default :obj:`None`)
    """
    import onnx

    # Silence exporter logs
    for name in ('onnx_ir', 'onnxscript'):
        logging.getLogger(name).setLevel(logging.WARNING)

    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*shares the same shape constraints with another axis.*")
        onnx_program = torch.onnx.export(
            wrapper, (), kwargs=inputs,
            input_names=list(inputs.keys()),
            output_names=wrapper.output_names,
            dynamic_shapes=wrapper.dynamic_shapes(inputs),
            dynamo=True,
            opset_version=opset_version,
            optimize=True,
            external_data=False,
            verbose=False)
    model = onnx_program.model_proto

    # Remove debugging metadata (stack traces) from nodes
    def strip(graph):
        for node in graph.node:
            del node.metadata_props[:]
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    strip(attr.g)
                for g in attr.graphs:
                    strip(g)
    strip(model.graph)

    # Name the neutrino dimension
    for output in model.graph.output:
        if output.name in ('nu_out', 'nu_batch'):
            output.type.tensor_type.shape.dim[0].dim_param = 'n_neutrinos'

    for key, value in (metadata or {}).items():
        model.metadata_props.add(key=key, value=str(value))

    onnx.checker.check_model(model)
    onnx.save(model, save_as)


def check_consistency(wrapper: VyPERONNX, samples: List[Dict[str, Tensor]], onnx_file: str,
                      rtol: float, atol: float) -> None:
    r"""Compare `onnxruntime` outputs with `pytorch` outputs. If the noise is
    not a model input, only the shape of `nu_out` is checked.

    Args:
        wrapper (VyPERONNX): the exported model.
        samples (List[Dict[str,Tensor]]): model inputs.
        onnx_file (str): the `.onnx` file.
        rtol (float): relative tolerance.
        atol (float): absolute tolerance.
    """
    import onnxruntime

    session = onnxruntime.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    for i, inputs in enumerate(samples):
        with torch.no_grad():
            pt_out = wrapper(**inputs)
        ort_out = session.run(None, {key: value.numpy(force=True) for key, value in inputs.items()})

        for name, pt, ort in zip(wrapper.output_names, pt_out, ort_out):
            ort = torch.from_numpy(ort)
            if name == 'nu_out' and 'noise' not in inputs:
                assert ort.shape == pt.shape and ort.isfinite().all(), \
                    f"Consistence check failed on output '{name}' (batch {i}): shape {tuple(ort.shape)}, expected {tuple(pt.shape)}."
                print(f"Checking output '{name}' (batch {i})... ✅ (shape only: noise is sampled inside the model)")
                continue
            torch.testing.assert_close(ort, pt, rtol=rtol, atol=atol,
                msg=lambda m: f"Consistence check failed on output '{name}' (batch {i}):\n{m}")
            print(f"Checking output '{name}' (batch {i})... ✅")


def benchmark(onnx_file: str, inputs: Dict[str, Tensor], repeat: int = 5) -> float:
    r"""Median `onnxruntime` latency (ms).

    Args:
        onnx_file (str): the `.onnx` file.
        inputs (Dict[str,Tensor]): model inputs.
        repeat (int, optional): number of runs. (default :obj:`int`=5)

    :rtype: :class:`float`
    """
    import onnxruntime

    session = onnxruntime.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    feeds = {key: value.numpy(force=True) for key, value in inputs.items()}
    session.run(None, feeds)
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        session.run(None, feeds)
        times.append(time.perf_counter() - start)
    return float(np.median(times) * 1e3)


def _git_commit() -> str:
    r"""Current git commit of VyPER, or `unknown`.

    :rtype: :class:`str`
    """
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                              cwd=osp.dirname(osp.abspath(__file__)), check=True).stdout.strip()
    except Exception:
        return 'unknown'


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def ONNX(cfg : DictConfig) -> None:
    r"""Perform VyPER model onnx export.

    Args:
        cfg (str): a `.yaml` file, stores training related parameters. (default: :obj:`str`=None).
    """
    print(OmegaConf.to_yaml(cfg))
    export_cfg = cfg['onnx_export']

    # Load checkpoint
    ckpt_file = ckpt_loader(cfg, key='onnx_export')

    # Load hyperparameters
    hparams_file = osp.join(export_cfg['model_directory'], "hparams.yaml")
    assert os.path.isfile(hparams_file), f"`hparams.yaml` is not found in {export_cfg['model_directory']}."

    assert cfg['datasets']['predict_set'] is not None, "Please provide an example dataset as `datasets.predict_set`."

    # Always use CPU for ONNX export
    print(f"Exporting model '{ckpt_file}' to ONNX.")
    model = VyPER.load_from_checkpoint(
        checkpoint_path = ckpt_file,
        hparams_file = hparams_file,
        map_location = torch.device('cpu'),
        num_sampling_steps = export_cfg['num_sampling_steps'],
        weights_only = False
    ).eval()
    wrapper = VyPERONNX(model).eval()

    # Example inputs, the first batch is used for export
    datamodule = VyPERDataModule(
        config = osp.join(hydra.utils.get_original_cwd(), f'configs/{HydraConfig.get().job.config_name}.yaml'),
        predict_set = cfg['datasets']['predict_set'],
        force_reload = cfg['datasets']['force_reload'],
        batch_size = 2, num_workers = 0, pin_memory = False)
    datamodule.setup("predict")
    samples = []
    for data in datamodule.predict_dataloader():
        samples.append(wrapper.example_inputs(data, export_cfg['num_sampling_steps'], export_cfg['noise_as_input']))
        if len(samples) == 2:
            break

    # Model metadata
    metadata = {
        'vyper.commit': _git_commit(),
        'vyper.checkpoint': Path(ckpt_file).name,
        'vyper.outputs': ', '.join(wrapper.output_names),
        'vyper.torch_version': torch.__version__,
    }
    if wrapper.use_diffusion:
        nu_cfg = cfg['target']['neutrinos']
        metadata.update({
            'vyper.num_sampling_steps': export_cfg['num_sampling_steps'],
            'vyper.noise_as_input': export_cfg['noise_as_input'],
            'vyper.noise_distribution': model.hparams.noise_distribution,
            'vyper.nu_features': ', '.join(nu_cfg['features']),
            'vyper.nu_reverse_transforms': ', '.join(nu_cfg['reverse_transforms']),
        })

    # Onnx export
    start = time.perf_counter()
    export_onnx(wrapper, samples[0], export_cfg['save_as'], export_cfg['opset_version'], metadata)
    print(f"ONNX export successful ({time.perf_counter() - start:.0f} s)! "
          f"ONNX model has been saved to '{export_cfg['save_as']}' "
          f"({os.path.getsize(export_cfg['save_as']) / 1e6:.1f} MB).")

    # Consistence check
    if export_cfg['consistence_check']['run_check']:
        rtol = export_cfg['consistence_check']['rtol']
        atol = export_cfg['consistence_check']['atol']
        print(f"Running consistence check, comparing `pytorch` outputs with `onnxruntime` outputs "
              f"with a relative tolerance of {rtol} and an absolute tolerance of {atol}.")
        check_consistency(wrapper, samples, export_cfg['save_as'], rtol, atol)
        print("ONNX consistence check successful!")

    # Inference speed
    n_graphs = int(samples[-1]['u'].size(0))
    print(f"ONNX Runtime (CPU) latency: {benchmark(export_cfg['save_as'], samples[-1]):.1f} ms "
          f"for a batch of {n_graphs} events.")


if __name__ == "__main__":
    # Required since PyTorch 2.6, see [#53](https://github.com/tzuhanchang/VyPER/pull/53).
    if version.parse(torch.__version__) >= version.parse("2.6"):
        torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr,torch_geometric.data.data.DataTensorAttr,torch_geometric.data.storage.GlobalStorage])
        torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig,omegaconf.base.ContainerMetadata,typing.Any,omegaconf.nodes.AnyNode,omegaconf.base.Metadata,omegaconf.listconfig.ListConfig,int])

    missing = [name for name, available in _REQUIREMENTS.items() if not available]
    if missing:
        raise ModuleNotFoundError(f"ONNX export requires {', '.join(f'`{name}`' for name in missing)} to be installed.")

    ONNX()
