# Environment

We provide three environment files, which can be used to setup Conda environment on [Linux](./environment-linux.yaml) or [MacOS](./environment-macos.yaml).
Currently, GPU acceleration is only available on Linux machines with CUDA&trade; devices.
MPS (Metal Performance Shaders) support will not be implemented for MacOS devices.

To setup Conda, please follow [miniforge install instruction](https://github.com/conda-forge/miniforge#install).

On a Linux-based machine:
```
conda env create -f environment-linux.yaml
```

For MacOS:
```
conda env create -f environment-maxos.yaml
```

On NVIDIA's Grace Hopper and Grace Blackwell machines:
```
conda env create -f environment-grace.yaml
```

After the Conda environment is successfully installed, to active the environment:
```
conda activate VyPER
```