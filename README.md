## Overview

Suite3D is a volumetric cell detection algorithm, generally applicable to any type of multi-plane functional 2p imaging where you see cells on multiple planes.
For an overview of the algorithms, [see our recent preprint](https://www.biorxiv.org/content/10.1101/2025.03.26.645628v1).

**[suite3d.github.io](https://suite3d.github.io)** has runnable demos, 3D visualizations, and step-by-step tutorials.

[Suite3D Viewer documentation](docs/gui.rst)

If you run into any kinks, please [open an issue](https://github.com/alihaydaroglu/suite3d/issues) and we'll be happy to help you get up and running.

## Installation

Suite3D needs Python 3.11 or 3.12.

```bash
pip install git+https://github.com/alihaydaroglu/suite3d.git
pip install 'cupy-cuda12x>=13.0,<14.0'    # GPU (registration); the CPU fallback is much slower
pip install 'suite3d[viz]'                # optional: napari 3D viewer
```

Registration runs on the GPU and needs a system [CUDA](https://developer.nvidia.com/cuda-downloads) 12.x install; keep `cupy` on the 13.x line. If you are not sure which CUDA you have, `conda install -c conda-forge cupy` will [sort it out for you](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-conda-forge).

To work on the source, clone and install it editable instead:

```bash
git clone https://github.com/alihaydaroglu/suite3d.git
cd suite3d
pip install -e ".[all]"
```

## Usage

Four worked demos live in [`demos/`](demos/), each on a real dataset:

| | demo | recording |
|---|---|---|
| **01** | [`01-v1-tc030/`](demos/01-v1-tc030/) | V1, standard 2P, 7 planes |
| **02** | [`02-lbm-ss004/`](demos/02-lbm-ss004/) | LBM, 22 planes |
| **03** | [`03-hippocampus/`](demos/03-hippocampus/) | CA1, standard 2P, 4 planes |
| **04** | [`04-sweep/`](demos/04-sweep/) | parameter sweep, reuses demo 03 |

Each demo runs two ways: a `run_pipeline.py` script that goes start to finish, or a `walkthrough.ipynb` notebook that steps through the pipeline one stage at a time. See [`demos/README.md`](demos/README.md) for the flags and data layout, or [suite3d.github.io](https://suite3d.github.io) to watch them run.

```bash
cd demos/01-v1-tc030
python run_pipeline.py --data-root /path/to/data --out-dir ./results
```

## Sample Data

The volumetric 2-photon datasets used to test Suite3D are on figshare:

**https://rdr.ucl.ac.uk/articles/dataset/Volumetric_2-photon_imaging_datasets_used_to_test_Suite3D/32956220**

Download and unpack it, then point each demo's `--data-root` at the folder.
