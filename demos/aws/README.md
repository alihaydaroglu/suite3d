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
| **02 lbm (SS004)** | `g5.8xlarge` (128 GB) as shipped, or `g5.4xlarge` (64 GB) with `--n-init-files 2 --t-batch-size 400` | **no** — never completed on EC2; the RAM figures in §1 are local measurements, and both EC2 attempts were OOM-killed on a 64 GB box |

> **Demo 02 has two heavy stages, and neither is registration.** Each was
> measured alone on this 22-plane recording:
>
> | stage | peak RAM | knob |
> |---|---:|---|
> | init, `n_init_files=1` | 30.6 GiB | `n_init_files` |
> | init, `n_init_files=2` | 58.2 GiB | |
> | init, **`n_init_files=4`** (shipped) | **113.5 GiB** | |
> | corrmap, **`t_batch_size=800`** (shipped) | **85.5 GiB** | `t_batch_size` |
> | corrmap, `t_batch_size=650` | 75.4 GiB | |
> | corrmap, `t_batch_size=400` | 41.5 GiB | |
> | corrmap, `t_batch_size=325` | 40.7 GiB | |
>
> **As shipped, the init pass is the ceiling at 113.5 GiB**, with the correlation
> map close behind at 85.5 GiB. Demo 02 therefore wants a **128 GB** box —
> `g5.8xlarge` (32 vCPU, 128 GB, A10G, ~$2.44/hr) — and even that has only ~14 GiB
> of headroom. On a 64 GB `g5.4xlarge` it dies about two minutes in at *"Applying
> plane alignment shifts"*; on a 32 GB box (`g4dn.2xlarge`, `g5.2xlarge`) it never
> gets close.
>
> Both knobs count **volumes/files, not bytes**, and the registered movie is
> float16 on disk but float32 in RAM.
>
> **To run demo 02 on a smaller box**, lower them on the command line:
> `--n-init-files 2 --t-batch-size 400` brings the two peaks to 58.2 and 41.5 GiB,
> which fits a 64 GB `g5.4xlarge` with under 4 GiB to spare. **Both flags change
> the result** — see the two warnings below. Neither is a dataset parameter, which
> is the point: you are trading science for hardware, and you should have to say so
> on the command line.
>
> ⚠ **`--n-init-files` changes the crosstalk you subtract.** On LBM the init pass
> *estimates the cavity crosstalk coefficient*, and `subtract_crosstalk=True` then
> removes it from the movie. Fewer tifs, noisier and smaller estimate: 0.080 /
> 0.125 / 0.155 for 1 / 2 / 4 files, against the reference run's 0.160. Lowering it
> does not make the run cheaper; it makes it different.
>
> ⚠ **`--t-batch-size` is not a free memory knob — it changes the science.**
> Lowering it shortens the temporal high-pass window (clamped to the batch),
> changes the running standard-deviation normalizer, and discards up to
> `detection_timebin - 1` volumes per batch. Measured on demo 02:
>
> | | `t_batch_size=800` | `t_batch_size=400` |
> |---|---:|---:|
> | corrmap peak RAM | 85.5 GiB | 41.5 GiB |
> | volumes used | 216 | 214 |
> | vmap correlation | — | 0.922 |
> | **ROIs found** | **40,608** | **48,587 (+19.6%)** |
>
> A correlation map that looks 92% the same yields **twenty percent more ROIs** —
> detection amplifies small changes in the map's fine structure. So a run with
> `--t-batch-size` will not reproduce the demo's reference segmentation, and you
> must not compare corrmaps or ROI counts across batch sizes. It is an escape
> hatch for fitting a smaller machine, not a tuning parameter — which is why it is
> a command-line flag and not one of the dataset's parameters.
>
> The GPU is never the constraint here: demo 02 uses ~8 GB of VRAM.

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
(TC030) on a `g4dn.2xlarge`: **25.4 min wall** for the full 19-tif, 4034-volume
dataset, and it fit. Demo 01 ships 10 of those tifs, so expect comfortably less.
At ~$0.75/hr that is well under a dollar of compute.

**Demo 02 (LBM) has never completed on EC2.** Everything below is measured
**locally** (RTX A4500, 128 GB host): 40,606–40,610 ROIs, 42 GB registered movie,
~8 GB VRAM, ~40 min wall. The only EC2 evidence is two OOM kills on a 64 GB
`g5.4xlarge` — one in the init pass at *"Applying plane alignment shifts"*, one in
the correlation map at *"Running batch 1 of 2"* — both consistent with the local
numbers. Treat the instance recommendation in §1 as a prediction, not a
measurement.

Peak host RAM as shipped is set by the **init pass**: 113.5 GiB (§1), with the
correlation map close behind at 85.5 GiB — which is why a 64 GB instance needs
both `--n-init-files 2` and `--t-batch-size 400`. Three things allocate heavily,
and they are easy to confuse:

* the **init pass**, which holds every init tif at once → scales with
  `n_init_files`; with the shipped settings this is the ceiling, 113.5 GiB.
* the **correlation map**, which processes `t_batch_size` volumes at a time and
  peaks at roughly twice one batch (movie plus its filtered copies); 85.5 GiB as
  shipped, a close second. Lowering it via `--t-batch-size` also shrinks the
  `mov_sub` chunks that segmentation later slices patches out of. But it changes
  the result — see the warning in §1.
* **trace extraction**, which loads `batchsize_frames` volumes of the *float32*
  registered movie and then duplicates them into shared memory. Unlike
  `t_batch_size`, this one *is* memory-only: extraction batches are independent,
  and `Fneu`/`spks` come out bit-identical when you change it. The demos size the
  batch to a whole multiple of the movie's on-disk chunk (100 volumes). A
  *smaller* batch is not automatically better: one that straddles a chunk
  boundary makes dask read two chunks (24.4 GiB / 9m40s) where a single aligned
  chunk reads one (22.4 GiB / 6m36s) — cheaper *and* faster.

**Segmentation is never the ceiling.** Its patches are sliced lazily out of a
dask array: 48 patches of 0.40 GB each on demo 02. Shrinking `patch_size_xy` to
save memory does nothing useful.

**Not measured.** Demo 03 on AWS: smaller and shallower than demo 01, should be
easier (locally, 5m24s).

Rough on-demand pricing, `us-east-1`, *check current rates*:

| instance | vCPU | RAM | GPU | ~$/hr |
|---|---:|---:|---|---:|
| `g4dn.2xlarge` | 8 | 32 GB | T4, 16 GB | 0.75 |
| `g5.2xlarge` | 8 | 32 GB | A10G, 24 GB | 1.21 |
| `g5.4xlarge` | 16 | 64 GB | A10G, 24 GB | 1.62 |
| `g5.8xlarge` | 32 | 128 GB | A10G, 24 GB | 2.44 |

EBS `gp3` is about $0.08/GB-month, so a 300 GB volume left running is roughly
$24/month whether or not the instance is on. **Delete the volume, not just the
instance.** Stopping an instance keeps billing its EBS.

## 6. If it breaks

**`CUDA_ERROR_INVALID_IMAGE`**, partway into registration — wrong cupy. Install
the 13.x line (§3).

**cupy imports but sees no GPU** — you are on a non-GPU instance, or the driver
is missing. `nvidia-smi` first; `bootstrap.sh` checks this for you.

**OOM a couple of minutes in, at "Applying plane alignment shifts".** That is the
**init pass**, and it is the most common way demo 02 dies. It holds every init
tif in RAM at once, so lower `n_init_files` (§1) or get a bigger box. Nothing
about registration or extraction is involved yet.

**OOM during registration.** If it is *GPU* memory, Suite3D auto-clamps
`gpu_reg_batchsize` to fit VRAM (`auto_adjust_batchsize=True`); if you still OOM,
lower `gpu_mem_safety_factor` (default `0.8`) before touching anything else.
Host-RAM OOM is a different problem: get a bigger instance.

**OOM during the correlation map**, at *"Running batch 1 of N"* or *"Binning with
timebin of size ..."*. Lower `t_batch_size` (§1) — but read the warning there
first: it changes the correlation map, so re-baseline your ROI count.

**OOM at the very end, after segmentation succeeded.** That is trace extraction.
Lower `--extract-batch-gb` (or set `--extract-batch` to one on-disk chunk, 100
volumes). Do not pick an arbitrary small number: a batch that straddles a chunk
boundary costs more memory *and* more time than one aligned chunk.

**Out of disk, mid-registration.** The registered movie is written next to the
raw data. See §2. This is the most common way these runs fail.

**napari errors over SSH** — use `--viewer none`.

**Host-RAM OOM in demo 02.** The two candidates are the init pass (~2 min in,
*"Applying plane alignment shifts"*) and the correlation map. Read the log line
you died on and turn the matching knob in §1; the peak is not wherever you
happen to be looking.

**`NameError: name 'psutil' is not defined`** during registration, in a container
or a bare venv — you have a build of suite3d from before `psutil` was declared a
dependency. `pip install psutil`.
</content>
