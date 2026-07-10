#!/usr/bin/env bash
# Provision a fresh AWS GPU instance to run the Suite3D demos, without Docker.
#
# Assumes Ubuntu 22.04 with an NVIDIA driver already present — i.e. the AWS
# "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)". On a bare
# Ubuntu AMI you must install the driver yourself first; that is the one step
# this script will not do for you.
#
#   curl -fsSL .../bootstrap.sh | bash          # or paste as EC2 user-data
#
# Idempotent: safe to re-run.

set -euo pipefail

REPO="${SUITE3D_REPO:-https://github.com/alihaydaroglu/suite3d.git}"
PREFIX="${SUITE3D_PREFIX:-$HOME/suite3d}"
VENV="$PREFIX/.venv"

echo "==> checking for an NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Use a GPU AMI with the driver preinstalled" >&2
    echo "       (Deep Learning Base OSS Nvidia Driver GPU AMI, Ubuntu 22.04)." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "==> apt deps"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3.11 python3.11-dev python3.11-venv git build-essential

echo "==> cloning suite3d into $PREFIX"
if [ -d "$PREFIX/.git" ]; then
    git -C "$PREFIX" pull --ff-only
else
    git clone --depth 1 "$REPO" "$PREFIX"
fi

echo "==> virtualenv"
python3.11 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel

# suite3d is NOT on PyPI, and its `gpu` extra is empty (all entries commented
# out in pyproject.toml), so cupy must be installed by hand. Pin to 13.x:
# cupy 14.x bundles NVRTC 13.0 and crashes on CUDA-12.x drivers with
# CUDA_ERROR_INVALID_IMAGE.
echo "==> cupy (pinned to the 13.x line)"
pip install 'cupy-cuda12x>=13.0,<14.0'

echo "==> suite3d (editable, from the clone)"
pip install -e "$PREFIX"

echo "==> verifying the GPU is visible to cupy"
python - <<'PY'
import cupy as cp
a = cp.arange(10)
assert int(a.sum()) == 45
dev = cp.cuda.Device(0)
free, total = dev.mem_info
print(f"  cupy OK — GPU 0, {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
PY

python -c "import suite3d, suite3d.job; print('  suite3d OK:', suite3d.__file__)"

cat <<EOF

Done. Activate and run a demo:

    source $VENV/bin/activate
    cd $PREFIX/demos/01-v1-tc030
    python run_pipeline.py --data-root /data/figshare --out-dir /data/results --viewer none

Use --viewer none on a headless instance. See demos/aws/README.md.
EOF
