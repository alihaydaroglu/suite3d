# Running demo 01 on AWS

Demo 01 (V1 / TC030) is the only demo run on EC2.

## Instance

`g4dn.2xlarge` (8 vCPU, 32 GB, T4). AMI: **Deep Learning Base OSS Nvidia Driver
GPU AMI (Ubuntu 22.04)**. EBS: 100 GB gp3.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/alihaydaroglu/suite3d/main/demos/aws/bootstrap.sh | bash
```

Installs Python 3.11, clones suite3d, creates a venv, installs cupy and suite3d,
and checks the GPU. Set `SUITE3D_REPO` to use a fork.

### Docker

```bash
# from the repo root
docker build -f demos/aws/Dockerfile -t suite3d-demos .

docker run --gpus all \
    -v /data/figshare:/data:ro \
    -v /data/results:/results \
    suite3d-demos \
    01-v1-tc030/run_pipeline.py --data-root /data --out-dir /results --viewer none
```

## Data

```bash
S3_BUCKET=my-suite3d-data ./fetch_data.sh v1 /data
```

Populate the bucket once from a machine that has the figshare archive:

```bash
aws s3 sync ./figshare s3://my-suite3d-data/ --exclude '*' --include 'v1/*'
```

Syncing from S3 needs an IAM instance profile on the instance.

## Run

```bash
source ~/suite3d/.venv/bin/activate
cd ~/suite3d/demos/01-v1-tc030
python run_pipeline.py --data-root /data --out-dir /opt/dlami/nvme/results --viewer none
```

Use `--out-dir` on the instance-store NVMe (`/opt/dlami/nvme`), not the EBS root.
Use `--viewer none` over SSH. Run under `tmux` or `nohup`.

Copy results back:

```bash
aws s3 sync /opt/dlami/nvme/results/s3d-results-demo-v1 s3://my-suite3d-data/results/demo-v1/
```

## Measured

`g4dn.2xlarge` (T4): 9m33s, 13.4 GiB peak RAM, 878 ROIs.

## Notes

- `pip install suite3d` and `suite3d[gpu]` do not work; install from git and add
  cupy by hand, pinned to `>=13.0,<14.0`.
- Delete the EBS volume when done, not just the instance.
