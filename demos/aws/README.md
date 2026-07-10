# Running the Suite3D demos on AWS

The demos need a CUDA GPU for registration and enough disk for the raw tifs plus
a registered movie of roughly the same size. A GPU EC2 instance handles all
three demos; this page is the recipe.

Nothing here is specific to AWS beyond the instance and storage choices — the
same steps work on any Ubuntu box with an NVIDIA driver.

---

## 1. Pick an instance

`g4dn.2xlarge` is the reference machine: 8 vCPU, 32 GB RAM, one T4 (16 GB
VRAM), ~$0.75/hr on-demand in `us-east-1`. It is what the published Suite3D
speed benchmark ran on.

| demo | instance | measured? |
|---|---|---|
| **01 v1 (TC030)** | `g4dn.2xlarge` | **yes** — see §5 |
| **03 hippocampus** | `g4dn.2xlarge` | no; smaller and shallower than demo 01, so it fits comfortably |
| **02 lbm (SS004)** | **≥128 GB RAM** (`g5.8xlarge`, `g6e.4xlarge`) | **RAM measured locally: 113.5 GiB peak.** Does **not** fit a 32 GB or 64 GB box — see §5. |

> ⚠ **Demo 02 needs ~114 GiB of RAM.** Measured end to end on one box
> (39m45s wall, 40,609 ROIs). A `g4dn.2xlarge`/`g5.2xlarge` (32 GB) or a
> `g5.4xlarge` (64 GB) will be OOM-killed during trace extraction, which is the
> peak — not during registration. Anything at or above 128 GB of host RAM works;
> the GPU only needs ~8 GB of VRAM. Note `g5.8xlarge` is 32 vCPU, so the default
> 8-vCPU G/VT quota will not cover it: request an increase first.

Two things that will trip you up:

* **vCPU quota.** A fresh AWS account often has a *"Running On-Demand G and VT
  instances"* limit of **8 vCPU**, which is exactly a `g4dn.2xlarge` or a
  `g5.2xlarge`. Anything larger (`g5.4xlarge` = 16 vCPU) needs a quota-increase
  request first, and those take a day or two to approve. Request it before you
  need it.
* **You do not need to tune `n_proc`.** Suite3D clamps its worker counts to
  `cpu_count() - 1`, so the shipped default of 16 becomes 7 on an 8-vCPU box.
  Leave it alone.

Use the **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)**. It
ships the NVIDIA driver, which is the one piece `bootstrap.sh` will not install
for you.

## 2. Size the disk

Registration writes a registered movie roughly as large as the raw data, and it
is written *alongside* it. Plan for at least `2 × raw`, plus room for the
correlation map and results.

| demo | raw | registered | EBS (gp3) |
|---|---:|---:|---:|
| 01 v1 | 21 GB | ~21 GB | **100 GB** |
| 03 hippocampus | 21 GB | ~21 GB | **100 GB** |
| 02 lbm | 57 GB | ~38 GB | **300 GB** |

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
clones suite3d, creates a venv, installs cupy and suite3d, and then *verifies
the GPU is actually visible to cupy* before exiting.

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

The demos never download anything themselves. Stage one dataset at a time:

```bash
S3_BUCKET=my-suite3d-data ./fetch_data.sh v1 /data
```

figshare has no stable per-file URLs, so `fetch_data.sh` cannot pull from it
unattended. Download the archive once, push the dataset folder to your own S3
bucket, and sync from there — inside the same region this is far faster than
re-downloading, and free. The script checks free space before it starts and
tells you what the layout must look like.

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
aws s3 sync /data/results/s3d-results-demo-v1 s3://my-suite3d-data/results/demo-v1/
# or: scp -r ubuntu@<instance>:/data/results/s3d-results-demo-v1 .
```

Run under `tmux` or `nohup`. Demo 02 takes hours, and an SSH drop will kill it.

## 5. What it costs, and what we actually measured

**Measured.** The Suite3D speed benchmark ran the *same recording as demo 01*
(TC030) on a `g4dn.2xlarge`: **25.4 min wall, 22 GB peak RAM** for the full
19-tif, 4034-volume dataset. Demo 01 ships 10 of those tifs, so expect
comfortably less. At ~$0.75/hr that is well under a dollar of compute.

Note how close 22 GB peak sits to the instance's 32 GB. Suite3D fits; it is not
swimming in headroom.

**Demo 02 (LBM), measured locally, never on AWS.** 39m45s wall, **113.5 GiB peak
RAM**, 40,609 ROIs, 42 GB registered movie, ~8 GB VRAM. The peak is in *trace
extraction*, not registration, so a box that survives registration can still be
OOM-killed at the very end. Pick ≥128 GB of host RAM. Do not try to tune around
an OOM here — the demo-01 numbers do not transfer.

**Not measured.** Demo 03 on AWS: smaller and shallower than demo 01, should be
easier (locally it is 5m24s / 8.1 GiB).

Rough on-demand pricing, `us-east-1`, *check current rates*:

| instance | vCPU | RAM | GPU | ~$/hr |
|---|---:|---:|---|---:|
| `g4dn.2xlarge` | 8 | 32 GB | T4, 16 GB | 0.75 |
| `g5.2xlarge` | 8 | 32 GB | A10G, 24 GB | 1.21 |
| `g5.4xlarge` | 16 | 64 GB | A10G, 24 GB | 1.62 |

EBS `gp3` is about $0.08/GB-month, so a 300 GB volume left running is roughly
$24/month whether or not the instance is on. **Delete the volume, not just the
instance.** Stopping an instance keeps billing its EBS.

## 6. If it breaks

**`CUDA_ERROR_INVALID_IMAGE`**, partway into registration — wrong cupy. Install
the 13.x line (§3).

**cupy imports but sees no GPU** — you are on a non-GPU instance, or the driver
is missing. `nvidia-smi` first; `bootstrap.sh` checks this for you.

**OOM during registration.** Suite3D auto-clamps `gpu_reg_batchsize` to fit
VRAM (`auto_adjust_batchsize=True`). If you still OOM, lower
`gpu_mem_safety_factor` (default `0.8`) before touching anything else. Host-RAM
OOM is a different problem: get a bigger instance.

**Out of disk, mid-registration.** The registered movie is written next to the
raw data. See §2. This is the most common way these runs fail.

**napari errors over SSH** — use `--viewer none`.

**OOM at the very end of demo 02**, after registration and segmentation both
succeeded — that is trace extraction hitting its ~114 GiB peak. Get a bigger box
(§1); there is nothing to tune.

**`NameError: name 'psutil' is not defined`** during registration, in a container
or a bare venv — you have a build of suite3d from before `psutil` was declared a
dependency. `pip install psutil`.
</content>
