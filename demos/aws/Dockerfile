# Suite3D GPU image for running the demos on an AWS GPU instance.
#
# Build from the root of the suite3d repo (the directory holding pyproject.toml):
#   docker build -f demos/aws/Dockerfile -t suite3d-demos .
#
# Derived from pub/fig-speed-comparison/Dockerfile.suite3d, which is the image
# the published AWS speed benchmark ran on.

# `devel`, not `runtime`: cupy 13.x's bundled NVRTC compiles kernels at import
# and needs CUDA headers (cuda_fp16.h) that only the devel image ships.
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN python -m pip install --upgrade pip setuptools wheel

# Pinned to the 13.x line on purpose. cupy 14.x bundles NVRTC 13.0, which emits
# kernels incompatible with CUDA-12.x host drivers (535.x etc.) and dies with
# CUDA_ERROR_INVALID_IMAGE on the first JIT-compiled kernel.
#
# Note: suite3d's `gpu` extra is currently empty, so `pip install suite3d[gpu]`
# installs no CUDA support at all. cupy must be installed explicitly, here.
RUN pip install 'cupy-cuda12x>=13.0,<14.0'

# Copy the whole repo: `pip install -e` needs pyproject.toml, which lives at the
# repo root, not inside the suite3d/ package directory.
COPY . /src/suite3d
RUN pip install -e /src/suite3d

WORKDIR /src/suite3d/demos

# Data and results are bind-mounted at run time; see demos/aws/README.md.
#   docker run --gpus all \
#     -v /data/figshare:/data:ro -v /data/results:/results \
#     suite3d-demos 01-v1-tc030/run_pipeline.py \
#       --data-root /data --out-dir /results --viewer none
ENTRYPOINT ["python"]
CMD ["01-v1-tc030/run_pipeline.py", "--help"]
