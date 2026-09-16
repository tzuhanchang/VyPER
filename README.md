# VyPER

<div align="center">

[![Workflows](https://github.com/tzuhanchang/VyPER/actions/workflows/testing.yml/badge.svg)](https://github.com/tzuhanchang/VyPER/actions/workflows/testing.yml)
[![Docs](https://img.shields.io/badge/Documentation-read-green.svg)](https://tzuhanchang.github.io/zensical-testing/)
[![Mattermost](https://img.shields.io/badge/Mattermost-join-0072C6?logo=mattermost&logoColor=white)](https://mattermost.web.cern.ch/hyper/channels/town-square)
[![Paper](https://img.shields.io/badge/Paper-2402.10149-b31b1b.svg)](https://arxiv.org/abs/2402.10149)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENCE)
<a href="https://huggingface.co/datasets/tzuhanchang/VyPER">
  <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg" height="20">
</a>

</div>

**VyPER** is a comprehensive event reconstruction algorithm for high-energy collider physics.

It is the next evolution of [HyPER](https://github.com/tzuhanchang/HyPER), which pioneered hypergraph representation learning for event reconstruction.
VyPER builds on this by robustly integrating generative modelling with message-passing, enabling full event reconstruction even for events with multiple neutrinos in the final state.

______________________________________________________________________

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Citation](#citation)

## Installation

Clone the repository:

```bash
git clone https://github.com/tzuhanchang/VyPER.git
```

Then create the conda environment using the file that matches your platform:

```bash
conda env create -f environment/environment-linux-amd64.yaml
```

VyPER supports the following platforms and compute backends:

|                 | CPU | CUDA |
| --------------- | --- | ---- |
| **linux-amd64** | ✅   | ✅   |
| **linux-arm64** | ✅   | ✅   |
| **macos**       | ✅   |     |  

Select the corresponding environment file from the `environment/` directory (e.g. `environment-linux-arm64.yaml`, `environment-macos.yaml`) when building environment.

## Quick Start

VyPER uses a specific HDF5 dataset structure to store event-level and object-level information required for reconstruction. To learn how to construct such a dataset, check out our [documentation](https://tzuhanchang.github.io/zensical-testing/), which walks through the expected schema and provides example scripts for converting your own data.

> If you'd rather skip dataset preparation and dive straight in, we also provide several pre-built datasets on [Hugging Face](https://huggingface.co/datasets/tzuhanchang/VyPER), ready to use out of the box.

To launch training, use the following command in the VyPER directory:

```bash
python -m VyPER.train --config-name=<ConfigFile>
```

where `<ConfigFile>` is the configuration file. Use `default` as `<ConfigFile>` to run with the [example configuration](configs/default.yaml).
A detailed configuration manual can be found [here](configs/README.md).

To make predictions with a trained model, run:

```bash
python -m VyPER.predict --config-name=<ConfigFile>
```

For more details and advanced settings, please refer to our [documentation](https://tzuhanchang.github.io/zensical-testing/dataset/hdf5/).

## Architecture

The backbone of VyPER is a multi-layer message-passing framework that explores and updates the graph latent space by exchanging information between neighboring graph objects.
Each message-passing layer leverages relational information encoded in the graph structure to construct neutrino latent representations and update edge, node, and global feature vectors.

<p align="center">
  <img height="350" src="https://raw.githubusercontent.com/tzuhanchang/github-artifacts/main/repositories/VyPER/images/network.svg" />
</p>

The updated node, edge, and global states from each layer are extracted and aggregated to perform edge classification and hyperedge classification, identifying the origins of each final-state object, while the constructed neutrino representations are used to condition the diffusion modelling, which ultimately predicts neutrino momentum.

## Citation

If you use VyPER in your analysis, please cite:

### HyPER

``` bibtex
@article{Birch-Sykes:2024gij,
    author = "Birch-Sykes, Callum and Le, Brian and Peters, Yvonne and Simpson, Ethan and Zhang, Zihan",
    title = "{Reconstructing short-lived particles using hypergraph representation learning}",
    eprint = "2402.10149",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1103/PhysRevD.111.032004",
    journal = "Phys. Rev. D",
    volume = "111",
    number = "3",
    pages = "032004",
    year = "2025"
}
```

This project was partially funded by the European Research Council (ERC), under Grant No. 817719.
