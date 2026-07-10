# Running demo 01 on AWS

This page covers **demo 01 (V1 / TC030)**, the only demo that has been run end to
end on EC2. It is a complete, verified recipe: instance, disk, provisioner,
data staging, and what it cost.

The other demos are not documented here on purpose. Demo 02 (LBM) needs ~114 GB
of host RAM and has never completed on EC2; demo 03 has never been attempted
there. Rather than print instance recommendations nobody has tested, this page
says nothing about them. If you want the experimental AWS notes for those, they
live on the `demo-dev` branch.

Nothing here is specific to AWS beyond the instance and storage choices — the
same steps work on any Ubuntu box with an NVIDIA driver.

---

## 1. Pick an instance

`g4dn.2xlarge` is the reference machine: 8 vCPU, 32 GB RAM, one T4 (16 GB VRAM),
~$0.75/hr on-demand in `us-east-1`. It is what the published Suite3D speed
benchmark ran on, and what demo 01 was measured on (§5).

Two things that will trip you up:

* **vCPU quota.** A fresh AWS account often has a *"Running On-Demand G and VT
  instances"* limit of **8 vCPU**, which is exactly a `g4dn.2xlarge`. Anything
  larger needs a quota-increase request first, and those take a day or two to
  approve. Request it before you need it.
* **You do not need to tune `n_proc`.** Suite3D clamps its worker counts to
  `cpu_count() - 1`, so the shipped default of 16 becomes 7 on an 8-vCPU box.
  Leave it alone.

Use the **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)**. It ships
the NVIDIA driver, which is the one piece `bootstrap.sh` will not install for
you.

## 2. Size the disk

Registration writes a registered movie roughly as large as the raw data, and it
is written *alongside* it. Plan for at least `2 × raw`, plus room for the
correlation map and results.

| demo | raw | registered | EBS (gp3) |
|---|---:|---:|---:|
| 01 v1 | 21 GB | ~21 GB | **100 GB** |

`g4dn` and `g5` instances also come with a local NVMe instance store (225 GB on
`g4dn.2xlarge`), mounted at `/opt/dlami/nvme` on the DLAMI. It is faster and
free, but **it is wiped when the instance stops**. Good for the job directory;
bad for anything you want to keep. If you use it, copy results off before you
stop the instance.

## 3. Install

```bash
# on the instance
curl -fsSL https://raw.githubusercontent.com/alihaydaroglu/suite3d/main/demos/aws/bootstrap.sh | bash
```

or clone the repo and run `demos/aws/bootstrap.sh`. It installs Python 3.11,
clones suite3d, creates a venv, installs cupy and suite3d, and then *verifies the
GPU is actually visible to cupy* before exiting. Set `SUITE3D_REPO` to point it
at a fork or a local checkout.

Three traps it exists to handle:

* **suite3d is not on PyPI.** `pip install suite3d` fails. Install from git.
* **`pip install suite3d[gpu]` installs nothing.** The `gpu` extra in
  `pyproject.toml` is currently empty — every entry is commented out. You must
  `pip install 'cupy-cuda12x>=13.0,<14.0'` yourself.
* The **cupy pin is not cosmetic.** cupy 14.x bundles NVRTC 13.0, which emits
  kernels that CUDA-12.x host drivers reject with `CUDA_ERROR_INVALID_IMAGE` the
  first time Suite3D JIT-compiles a kernel — i.e. partway into registration,
  after you have already paid for the data transfer.

### Docker instead

```bash
# from the repo root, so the build context includes suite3d/
docker build -f demos/aws/Dockerfile -t suite3d-demos .

docker run --gpus all \
    -v /data/figshare:/data:ro \
    -v /data/results:/results \
    suite3d-demos \
    01-v1-tc030/run_pipeline.py --data-root /data --out-dir /results --viewer none
```

Needs the NVIDIA container toolkit (preinstalled on the DLAMI).

## 4. Get the data onto the instance

The demos never download anything themselves.

```bash
S3_BUCKET=my-suite3d-data ./fetch_data.sh v1 /data
```

figshare has no stable per-file URLs, so `fetch_data.sh` cannot pull from it
unattended. Download the archive once, push the dataset folder to your own S3
bucket, and sync from there — inside the same region this is far faster than
re-downloading, and free. The script checks free space before it starts and tells
you what the layout must look like. Syncing from S3 needs an **IAM instance
profile** on the instance; a plain `aws s3 sync` from an unprivileged box will
fail.

Then run:

```bash
source ~/suite3d/.venv/bin/activate
cd ~/suite3d/demos/01-v1-tc030
python run_pipeline.py --data-root /data --out-dir /opt/dlami/nvme/results --viewer none
```

> ⚠ **Put `--out-dir` on the instance-store NVMe, not on the EBS root.**
> Registration is I/O-bound, not GPU-bound: writing the job directory to a
> default `gp3` root volume roughly **doubles** it (@benchmark_tester measured
> 2125 s vs 410 s on the same instance for the same data; CPU-seconds are
> identical either way, the EBS run just stalls at ~12 % CPU). The NVMe at
> `/opt/dlami/nvme` is faster and free — but it is **wiped when the instance
> stops**, so copy results off before you stop it (§2).

**Use `--viewer none` on the instance.** `--viewer napari` needs a display and
will fail over SSH. Copy the exported results directory back to your laptop and
view it there:

```bash
aws s3 sync /opt/dlami/nvme/results/s3d-results-demo-v1 s3://my-suite3d-data/results/demo-v1/
# or: scp -r ubuntu@<instance>:/opt/dlami/nvme/results/s3d-results-demo-v1 .
```

Run under `tmux` or `nohup`; an SSH drop will kill the run.

## 5. What it costs, and what we actually measured

**Demo 01, measured on a `g4dn.2xlarge` (T4).** The full demo, all five stages:

| | |
|---|---:|
| wall clock | **9m 33s** |
| peak host RAM | 13.4 GiB |
| ROIs | 878 |
| init / register / corrmap / segment / extract | 83 s / 272 s / 49 s / 78 s / 54 s |

At ~$0.75/hr that is about **$0.12 of compute**, plus data transfer and the EBS
volume. The same recording, at its full 19-tif / 4034-volume length, took 25.4
min in the published Suite3D speed benchmark on the same instance type; demo 01
ships 10 of those tifs.

878 ROIs against 885 locally — a 0.8% difference that comes from GPU and BLAS
arithmetic, not randomness in the algorithm. See `demos/README.md`.

Rough on-demand pricing, `us-east-1`, *check current rates*:

| instance | vCPU | RAM | GPU | ~$/hr |
|---|---:|---:|---|---:|
| `g4dn.2xlarge` | 8 | 32 GB | T4, 16 GB | 0.75 |

EBS `gp3` is about $0.08/GB-month, so a 100 GB volume left lying around is
roughly $8/month whether or not the instance is on. **Delete the volume, not just
the instance.** Stopping an instance keeps billing its EBS.

## 6. If it breaks

**`CUDA_ERROR_INVALID_IMAGE`**, partway into registration — wrong cupy. Install
the 13.x line (§3).

**cupy imports but sees no GPU** — you are on a non-GPU instance, or the driver
is missing. `nvidia-smi` first; `bootstrap.sh` checks this for you.

**`NameError: name 'psutil' is not defined`** during registration, in a container
or a bare venv — you have a build of suite3d from before `psutil` was declared a
dependency. `pip install psutil`.

**Out of disk, mid-registration.** The registered movie is written next to the
raw data. See §2. This is the most common way these runs fail.

**OOM during registration.** If it is *GPU* memory, Suite3D auto-clamps
`gpu_reg_batchsize` to fit VRAM (`auto_adjust_batchsize=True`); if you still OOM,
lower `gpu_mem_safety_factor` (default `0.8`) before touching anything else.
Host-RAM OOM is a different problem: get a bigger instance.

**OOM at the very end, after segmentation succeeded.** That is trace extraction.
Lower `--extract-batch-gb` (or set `--extract-batch` to one on-disk chunk, 100
volumes). Do not pick an arbitrary small number: a batch that straddles a chunk
boundary costs more memory *and* more time than one aligned chunk. Extraction
batches are independent, so this changes memory and nothing else.

**napari errors over SSH** — use `--viewer none`.
