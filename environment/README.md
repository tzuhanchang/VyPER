# Environment

We provide three environment files, which can be used to setup Conda environment on [Linux-amd64](./environment-linux-amd64.yaml) or [Linux-arm64](./environment-linux-arm64.yaml) or [MacOS](./environment-macos.yaml).
Currently, GPU acceleration is only available on Linux machines with CUDA&trade; devices.
MPS (Metal Performance Shaders) support will not be implemented for MacOS devices.

To setup Conda, please follow [miniforge install instruction](https://github.com/conda-forge/miniforge#install).

On a Linux-based machine with amd64 architecture:
```
conda env create -f environment-linux-amd64.yaml
```

On a Linux-based machine with arm64 architecture:
```
conda env create -f environment-linux-arm64.yaml
```

For MacOS:
```
conda env create -f environment-maxos.yaml
```

After the Conda environment is successfully installed, to active the environment:
```
conda activate VyPER
```

Alternatively, we provide Docker images for both Linux-amd64 and Linux-arm64 on [GitHub Container Registry](https://github.com/tzuhanchang/VyPER/pkgs/container/vyper). They are automatically updated whenever any changes are made to the base environment files.