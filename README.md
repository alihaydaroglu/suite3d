## Overview

Suite3D is a volumetric cell detection algorithm, generally applicable to any type of multi-plane functional 2p imaging where you see cells on multiple planes.
For an overview of the algorithms, [see our recent preprint](https://www.biorxiv.org/content/10.1101/2025.03.26.645628v1).

You might run into few kinks - please reach out to Ali (ali.haydaroglu.20@ucl.ac.uk, or by creating issues on this repository) and I'll be happy to help you get up and running.

## Installation

``` bash
git clone git@github.com:alihaydaroglu/suite3d.git
cd suite3d
```

### Option 1: `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package manager. Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate  # linux/macOS
# .venv\Scripts\activate   # windows

# Core compute only
uv pip install -e .

# With napari visualization and jupyter
uv pip install -e ".[viz,jupyter]"

# Everything (viz, jupyter, webui, etc.)
uv pip install -e ".[all]"

# Web UI only (for a monitoring-only machine)
uv pip install -e ".[webui]"
```

### Option 2: `conda` (miniforge3 only)

``` bash
conda create -n s3d -c conda-forge python=3.11
conda activate s3d
pip install -e ".[all]"
```

Or use the provided environment file:
```bash
conda env create -f environment.yml
conda activate s3d
pip install -e ".[all]"
```

### Option 3: `pip`

``` bash
python -m venv .venv
source .venv/bin/activate      # linux, macOS
# .venv\Scripts\activate       # windows

pip install -e ".[all]"  # include all optional dependencies
```


### GPU Dependencies

To use the GPU, you need a system [`cuda`](https://developer.nvidia.com/cuda-downloads) installation.
We recommend `12.x`.

After downloading CUDA, use the corresponding pip install for cupy:

| Supported CUDA Toolkits: v11.2 / v11.3 / v11.4 / v11.5 / v11.6 / v11.7 / v11.8 / v12.0 / v12.1 / v12.2 / v12.3 / v12.4 / v12.5 / v12.6 / v12.8

```bash
pip install cupy-cuda12x  # or 11x if you installed CUDA v11.2 - v11.8
```

If you are unsure what CUDA toolkit you have installed, you can install `cupy` through `conda` and it will [handle the CUDA requirements for you](https://docs.cupy.dev/en/v12.2.0/install.html#installing-cupy-from-conda-forge):
```bash
conda install -c conda-forge cupy
```


**Note on `conda` environments**
We highly recommend switching from your current conda package manager to miniforge3 if you have not yet done so. If not on miniforge3, and the installation gets stuck around "Solving Environment", you should use libmamba ([explanation](https://conda.github.io/conda-libmamba-solver/libmamba-vs-classic/)), install it using the [instructions here](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community). Also, set the conda channel priority to be strict: `conda config --set channel_priority strict`. It's important that you don't forget the `-e` in the pip command, this allows the installation to be editable.

## Usage

### Notebooks

Run a jupyter notebook in this environment, either by running `jupyter notebook` in the activated environment or running a jupyter server from a different conda env and selecting this environment for the kernel ([see here](https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084)). Make sure you use the correct environment!

Then, run the Demo notebook in `demos/`.

### Scripts

Standalone Python scripts for running the pipeline are provided in `demos/`:
- `demos/demo_standard_2p.py` - Full pipeline for standard 2-photon data
- `demos/demo_lbm.py` - Full pipeline for Light Beads Microscopy data

Run them with:
```bash
python demos/demo_standard_2p.py --data_dir /path/to/tifs --output_dir /path/to/output
```

### Web UI (Job Monitoring)

The web UI lets you monitor suite3d jobs (view initialization results, registration quality, correlation maps, and curate cells). It runs in a separate process and reads job files from disk.

Install the webui dependencies:
```bash
pip install -e ".[webui]"
```

Launch the web UI:
```bash
panel serve serve_webui.py --show
```

## Docker

There is a Dockerfile in this repo that successfully builds (`docker build - < Dockerfile`). Contributions to get this running with napari in a container are welcome.

## Sample Data
Use [this](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucqfhay_ucl_ac_uk/EuQX2PFw13xHhILvRux29AQB48tXCxBJQ7z6JfHee25pfw?e=HmBlAc) for the standard 2p imaging demo, recorded in mouse CA1, courtesy of Andrew Landau.

Sample LBM data coming soon!
