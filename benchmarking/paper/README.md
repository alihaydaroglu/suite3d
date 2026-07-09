# Speed benchmark: Suite3D, CaImAn (offline CNMF) and CaImAn OnACID

The exact code used to produce the paper's speed benchmark, on AWS.

Registration and detection are timed as separate phases. Each runner writes one
CSV row per phase (wall time, CPU%, peak resident memory) plus a `.npz` of the
0.5 s resource trace.

| file | purpose |
|---|---|
| `config.py` | dataset path and the locked parameter sets for both tools |
| `timing_harness.py` | per-phase wall/CPU/memory recorder |
| `run_suite3d.py` | Suite3D: registration + detection |
| `run_caiman.py` | CaImAn: NoRMCorre, then offline CNMF and/or OnACID |
| `run_caiman_register.py` | NoRMCorre only; writes the memmap path to `/results/memmap.txt` |
| `run_caiman_onacid.py` | OnACID only, reading an existing memmap |
| `Dockerfile.suite3d` | GPU image (CUDA 12.2, cupy 13.x, suite3d) |
| `Dockerfile.caiman` | CPU image (caiman 1.12.2) |

## 1. Data

The recording is `TC030_2025-03-25`, experiment 3, published as the `v1/` folder
of the figshare deposit:

> **figshare DOI: `<TO BE FILLED IN ON PUBLICATION>`**

Download `v1/raw/` — 10 ScanImage TIFF files, 21.1 GB. The benchmark runs on the
first **2** or the first **10** of them, selected with `--subset`.

Place them so the container sees them at
`/data/raw/TC030/2025-03-25/3/*.tif` (see the `-v` flags below). `config.py`
resolves that path; nothing else reads the data.

## 2. Instances

| tool | instance | why |
|---|---|---|
| Suite3D | `g4dn.2xlarge` (8 vCPU, 30 GB, Tesla T4) | uses the GPU for registration |
| CaImAn / OnACID | `r6a.2xlarge` (8 vCPU, 64 GiB, no GPU) | CaImAn has no GPU path; needs the memory |

Both runs are capped at 8 CPUs (`--cpus=8`) so the two instances are matched on
core count. `--shm-size=8g` is required for Suite3D: it uses `/dev/shm` to pass
movie batches to detection workers, and Docker's 64 MB default causes a SIGBUS.

## 3. Build

From the repository root (the directory holding `pyproject.toml`):

```bash
docker build -f benchmarking/paper/Dockerfile.suite3d -t s3d-bench:suite3d .
docker build -f benchmarking/paper/Dockerfile.caiman  -t s3d-bench:caiman  .
```

## 4. Run

Set `DATA` to the directory holding the TIFFs, and make `scratch/` and
`results/` writable.

```bash
export DATA=/path/to/v1/raw
mkdir -p scratch results

MOUNTS="-v $DATA:/data/raw/TC030/2025-03-25/3:ro \
        -v $PWD/scratch:/scratch \
        -v $PWD/results:/results"
```

**Suite3D** (on `g4dn.2xlarge`). `--subset` selects the first N TIFFs.

```bash
docker run --rm --gpus all --cpus=8 --shm-size=8g $MOUNTS \
  -e INSTANCE_LABEL=g4dn.2xlarge \
  s3d-bench:suite3d --subset 2 --n-cores 8
```

**CaImAn** (on `r6a.2xlarge`). `--mode` selects the detection path.

```bash
# offline CNMF (fit + evaluate_components + refit)
docker run --rm --cpus=8 --shm-size=8g $MOUNTS \
  -e INSTANCE_LABEL=r6a.2xlarge \
  s3d-bench:caiman --subset 2 --n-cores 8 --mode offline
```

**OnACID** (on `r6a.2xlarge`). Register once, then run OnACID against the
resulting memmap, so registration is not timed twice:

```bash
docker run --rm --cpus=8 --shm-size=8g $MOUNTS \
  -e INSTANCE_LABEL=r6a.2xlarge --entrypoint python \
  s3d-bench:caiman /work/run_caiman_register.py --subset 2 --n-cores 8

docker run --rm --cpus=8 --shm-size=8g $MOUNTS \
  -e INSTANCE_LABEL=r6a.2xlarge --entrypoint python \
  s3d-bench:caiman /work/run_caiman_onacid.py --subset 2 --n-cores 8 \
    --memmap-path-file /results/memmap.txt
```

Replace `--subset 2` with `--subset 10` for the longer run.

## 5. Parameters

All parameters live in `config.py` and are applied by every runner. Notably
`K = 1000` and `pw_rigid = True` for CaImAn; Suite3D runs 3D non-rigid
registration on the GPU and is given the 7 functional planes (2–8), while CaImAn
is given all 9 acquired planes.

`run_caiman_onacid.py` sets `p = 0`. CaImAn 1.12.2 cannot run OnACID on 3D data
otherwise: `CNMF.fit` leaves `estimates.g/lam/bl/neurons_sn` as `None` for 3D,
and `OnACID._prepare_object` iterates over them. The `p = 0` branch skips that
code. With `p = 0` OnACID performs no deconvolution and returns no spikes.

## 6. Output

Written to `results/`:

- `timings.csv` — one row per phase: `tool, phase, instance, dataset, subset,
  wall_s, avg_cpu_pct, peak_cpu_pct, avg_mem_gb, peak_mem_gb, n_cpu_cores`
- `samples_<tool>_<phase>[_subsetN].npz` — 0.5 s trace of `t_s, cpu_pct, mem_gb`
- `memmap.txt` — path to the CaImAn memmap, when registration is run separately
- Suite3D job directory and CaImAn `results.hdf5` under `scratch/` and `results/`

`INSTANCE_LABEL` (or `--instance`) only labels the `instance` column; it defaults
to `local`.

CPU percentages are core-equivalents: 800% means eight saturated vCPUs.
