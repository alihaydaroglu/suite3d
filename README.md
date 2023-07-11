## Installation
```
git clone --recurse-submodules git@github.com:alihaydaroglu/s2p-lbm.git
conda env create -f environment.yml
conda activate s2p-lbm
cd suite2p
pip install -e .
```

## Usage
Run a jupyter notebook in this envinronment, either by installing jupyter in the s2p-lbm environment, or running a jupyter server from a different conda env and selecting this environment for the kernel (https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084).

Then, run the Demo notebook.


## UI only installation
If you want to only visualize results on your local laptop, you can install a lightweight script that doesn't have any of the computational dependencies. 

```
git clone --recurse-submodules git@github.com:alihaydaroglu/s2p-lbm.git
conda create -y -n s3d-vis -c conda-forge python=3.9
conda activate s3d-vis
pip install "napari[pyqt5]"
pip install notebook
conda install pyqtgraph
conda install -c conda-forge matplotlib
```

## Docker

There is a Dockerfile in this repo that successfully builds (`docker build - < Dockerfile`). I don't know anything about Docker, but I would love to have this successfully run in a container. If you manage to get that working let me know! Ideally, this would also include some sort of X host to run napari (https://napari.org/stable/howtos/docker.html#base-napari-image), presumably there is a way to merge the napari-xpra docker image into this one to make that work. 

## Sample Data
Here's a sample tiff file to test the code on. If you can't access the dropbox let me know, we might need to get Alipasha to add you to the folder. https://www.dropbox.com/sh/qp1otwnipiufjqz/AABf3iEq5ggAUVh0P_WwnuSPa?dl=0
