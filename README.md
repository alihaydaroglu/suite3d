## Installation

Clone this repository, and the suite2p submodule. To set up the submodule, run `git submodule init` and then `git submodule update`.

Install the verison of suite2p in the submodule in developer mode in a *fresh* conda environment. You might need to manually install mkl_fft, PyQt5 etc. from conda.

Install napari in this environment (either through pip or conda)

You need a way to run jupyter notebooks in this conda environment. My suggestion is to install `ipykernel` with conda (`conda install ipykernel`). I usually launch my notebook server from my base conda environment, and all of my environments with ipykernel installed will show up as options when creating/running a notebook. This way I can have many notebooks using different environments on the same notebook server. You want to run this notebook on a powerful computer with a lot of RAM (128 GB works for me). You can make this notebook available to other computers on your local subnet (e.g. your laptop in the lab or connected to VPN) when launching the notebook server by specifying an IP and port (`jupyter notebook --ip 128.xxx.xxx.xxx --port yyyy`) and then connecting to this address from your personal computer (`128.xxx.xxx.xxx:yyyy`).


These instructions are not thoroughly tested and are probably not foolproof, if something goes wrong, try to manually install the missing package, google the error, or restart with a fresh environment and try again.

### Example installation
```
git clone --recurse-submodules git@github.com:alihaydaroglu/s2p-lbm.git
cd suite2p
conda config --append channels conda-forge
conda create -n s2p-lbm
conda activate s2p-lbm
conda install python=3.8 pip
pip install -e .
conda install mkl numba numpy=2.13.5 h5py=2.10.0 tbb scikit-learn matplotlib ipykernel napari
conda install mkl_fft
conda install imreg_dft
conda install tensorflow
conda install dask-image
conda install mrcfile
```

### Sample Data
Here's a sample tiff file to test the code on. If you can't access the dropbox let me know, we might need to get Alipasha to add you to the folder. https://www.dropbox.com/sh/qp1otwnipiufjqz/AABf3iEq5ggAUVh0P_WwnuSPa?dl=0
