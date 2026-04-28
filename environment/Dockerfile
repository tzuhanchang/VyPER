FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04

ARG TARGETOS
ARG TARGETARCH

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Update package list and install essential packages
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install miniforge
RUN curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -o miniforge.sh \
    && bash miniforge.sh -b -p /opt/miniforge \
    && rm miniforge.sh

# Add miniforge to PATH
ENV PATH="/opt/miniforge/bin:$PATH"

# Create a working directory
WORKDIR /workspace

# Copy the environment file
COPY environment-${TARGETOS}-${TARGETARCH}.yaml ./

# Create the conda environment from the environment file
RUN mamba env create -f environment-${TARGETOS}-${TARGETARCH}.yaml -y && \
    rm environment-${TARGETOS}-${TARGETARCH}.yaml

# Make sure conda is initialized for the shell
RUN mamba shell init -s bash

# Activate the environment and make it default
RUN echo "mamba activate VyPER" >> ~/.bashrc

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

# Set the default command to start a bash shell
CMD ["/bin/bash"]
