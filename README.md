## Overview

Suite3D is a volumetric cell detection algorithm, generally applicable to any type of multi-plane functional 2p imaging where you see cells on multiple planes.
For an overview of the algorithms, [see our recent preprint](https://www.biorxiv.org/content/10.1101/2025.03.26.645628v1).

You might run into few kinks - please reach out to Ali (ali.haydaroglu.20@ucl.ac.uk, or by creating issues on this repository) and I'll be happy to help you get up and running. 

## Installation

``` bash
git clone git@github.com:alihaydaroglu/suite3d.git
cd suite3d
```

`conda` (miniforge3 only)
``` bash
conda create -n s3d -c conda-forge python=3.11
conda activate s3d
pip install -e ".[all]"  # [all] optional
```


`pip`
``` bash
python -m venv
source .venv/bin/activate      # linux, macOS
# or
# source .venv/Scripts/activate  # windows

pip install ".[all]" % include viz/jupyter utilities
```


### GPU Dependencies

To use the GPU, you need a system [`cuda`](https://developer.nvidia.com/cuda-downloads) installation.
We recommend `12.x`.

After downloading CUDA, use the corresponding pip install for cupy:

| Supported CUDA Toolkits: v11.2 / v11.3 / v11.4 / v11.5 / v11.6 / v11.7 / v11.8 / v12.0 / v12.1 / v12.2 / v12.3 / v12.4 / v12.5 / v12.6 / v12.8

```bash
pip install cupy-cuda12x  # or 11x if you installed CUDA v11.2 - v11.8
```

If you are unsure what CUDA toolkit you have installed, you can install `cupy` through `conda` and it will [handle the CUDA requirements for you](see here: https://docs.cupy.dev/en/v12.2.0/install.html#installing-cupy-from-conda-forge):
```bash
conda install -c conda-forge cupy
```


**Note on `conda` environments**
We highly recommend switching from your current conda package manager to miniforge3 if you have not yet done so. If not on miniforge3, and the installation gets stuck around "Solving Environment", you should use libmamba ([explanation](https://conda.github.io/conda-libmamba-solver/libmamba-vs-classic/)), install it using the [instructions here](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community). Also, set the conda channel priority to be strict: `conda config --set channel_priority strict`. It's important that you don't forget the `-e` in the pip command, this allows the suite2p installation to be editable.

## Usage
Run a jupyter notebook in this envinronment, either by running `jupyter notebook` in the activated environment or running a jupyter server from a different conda env and selecting this environment for the kernel ([see here](https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084)). Make sure you use the correct environment!

Then, run the Demo notebook.

## Docker

There is a Dockerfile in this repo that successfully builds (`docker build - < Dockerfile`). I don't know anything about Docker, but I would love to have this successfully run in a container. If you manage to get that working let me know! Ideally, this would also include some sort of X host to run napari (https://napari.org/stable/howtos/docker.html#base-napari-image), presumably there is a way to merge the napari-xpra docker image into this one to make that work. 

## Sample Data
Use [this](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucqfhay_ucl_ac_uk/EuQX2PFw13xHhILvRux29AQB48tXCxBJQ7z6JfHee25pfw?e=HmBlAc) for the standard 2p imaging demo, recorded in mouse CA1, courtesy of Andrew Landau. 

Sample LBM data coming soon!

