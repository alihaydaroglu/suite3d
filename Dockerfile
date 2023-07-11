# Base image
FROM continuumio/miniconda3

# Metadata
LABEL maintainer='Ali Haydaroglu <ali.haydaroglu@outlook.com>'

# Create conda environment with Python 3.8
RUN conda create --name s2p-lbm python=3.8

# Activate the Conda Environment
SHELL ["conda", "run", "-n", "s2p-lbm", "/bin/bash", "-c"]

RUN conda install -c conda-forge git pip mkl tbb numpy numba napari matplotlib scikit-learn mrcfile mkl_fft dask-image
RUN pip install scipy>=1.4.0 torch>=1.7.1 natsort rastermap>0.1.0 tifffile scanimage-tiff-reader>=1.4.1 importlib-metadata paramiko pynwb sbxreader imreg-dft-nw


# Set the working directory
WORKDIR /app