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
| `run_caiman.py` | CaImAn: NoRMCorre, then offline CNMF |
| `run_caiman_register.py` | NoRMCorre only; writes the memmap path to `/results/memmap.txt` |
| `run_caiman_onacid.py` | OnACID only, reading an existing memmap |
| `Dockerfile.suite3d` | GPU image (CUDA 12.2, cupy 13.x, suite3d) |
| `Dockerfile.caiman` | CPU image (caiman 1.12.2) |

> ⚠ **Read [§7 Gotchas](#7-gotchas) before you start.** Two of them (storage for
> `scratch/`, and the offline-CNMF memory ceiling) will silently give you wrong
> numbers or an OOM kill if you skip them.

## 0. Get the code

`benchmarking/paper/` currently lives on the **`experimental`** branch, not on
the default branch:

```bash
git clone -b experimental https://github.com/alihaydaroglu/suite3d.git
cd suite3d
```

A plain `git clone` lands you on `main`, where this directory does not exist.

## 1. Data

The recording is `TC030_2025-03-25`, experiment 3, published as the `v1/` folder
of the figshare deposit:

> **figshare DOI: `<TO BE FILLED IN ON PUBLICATION>`**

Download `v1/raw/` — 10 ScanImage TIFF files, 21.1 GB. The benchmark runs on the
first **2** or the first **10** of them, selected with `--subset`.

Place them so the container sees them at
`/data/raw/TC030/2025-03-25/3/*.tif` (see the `-v` flags below). `config.py`
resolves that path; nothing else reads the data.

Both runners select `--subset N` as the first N files in lexicographic order, so
the two tools see the same data.

## 2. Instances and storage

| tool | instance | why |
|---|---|---|
| Suite3D | `g4dn.2xlarge` (8 vCPU, 30 GB, Tesla T4) | uses the GPU for registration |
| CaImAn / OnACID | `r6a.2xlarge` (8 vCPU, 64 GiB, no GPU) | CaImAn has no GPU path; needs the memory |

Both runs are capped at 8 CPUs (`--cpus=8`) so the two instances are matched on
core count. `--shm-size=8g` is required for Suite3D: it uses `/dev/shm` to pass
movie batches to detection workers, and Docker's 64 MB default causes a SIGBUS.

### Storage — this changes the answer

**Suite3D registration is I/O-bound.** Put the TIFFs *and* `scratch/` on fast
local storage, not on a default EBS root volume. `g4dn` instances ship a local
NVMe instance store; on the Deep Learning AMIs it is already mounted at
`/opt/dlami/nvme`. Measured on `g4dn.2xlarge`, subset 2, everything else identical:

| storage for data + scratch | registration | avg CPU |
|---|---|---|
| EBS gp3 root volume (111 MB/s) | 435 s | 51 % |
| instance-store NVMe (181 MB/s) | **206 s** | **104 %** |

Same CPU-seconds either way — the EBS run just spends half its wall time stalled
(median CPU 12 %). At `--subset 10` the gap is larger still: 2125 s on EBS vs
**410 s** on NVMe. So a benchmark run on slow storage overstates Suite3D's
registration by ~2–5× and quietly corrupts the Suite3D-vs-CaImAn comparison.

CaImAn's registration is CPU-bound, not I/O-bound: on `r6a` (EBS only, no
instance store) it lands at 655 s, matching the reference runs. EBS is fine there.

## 3. Build

From the repository root (the directory holding `pyproject.toml`):

```bash
docker build -f benchmarking/paper/Dockerfile.suite3d -t s3d-bench:suite3d .
docker build -f benchmarking/paper/Dockerfile.caiman  -t s3d-bench:caiman  .
```

Both Dockerfiles `COPY benchmarking/paper/ /work/`, so the repo's `.dockerignore`
must *not* exclude that directory. It carries an explicit `!benchmarking/paper/`
negation for exactly this reason — if you see

```
ERROR: failed to compute cache key: "/benchmarking/paper": not found
```

something has re-excluded it. (In `Dockerfile.caiman` this only surfaces *after*
the multi-minute mamba solve.)

The suite3d image is ~15 GB, the caiman image ~5.8 GB. Budget disk accordingly:
~150 GB for the GPU box, ~300 GB for the CPU box at `--subset 10`.

## 4. Run

Set `DATA` to the directory holding the TIFFs, and make `scratch/` and
`results/` writable. On the GPU box, put all three on the instance store (§2).

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

**CaImAn offline CNMF** (on `r6a.2xlarge`).

```bash
docker run --rm --cpus=8 --shm-size=8g $MOUNTS \
  -e INSTANCE_LABEL=r6a.2xlarge \
  s3d-bench:caiman --subset 2 --n-cores 8 --mode offline
```

`--mode` defaults to `offline`. **Do not pass `--mode onacid` or `--mode both`**:
they run OnACID with `p = 2`, which is exactly the 3D crash that
`run_caiman_onacid.py` exists to work around (see §5). Use the two-step recipe
below instead.

**OnACID** (on `r6a.2xlarge`). Register once, then run OnACID against the
resulting memmap, so registration is not timed twice *within the OnACID arm*:

```bash
docker run --rm --cpus=8 --shm-size=8g $MOUNTS \
  -e INSTANCE_LABEL=r6a.2xlarge --entrypoint python \
  s3d-bench:caiman /work/run_caiman_register.py --subset 2 --n-cores 8

docker run --rm --cpus=8 --shm-size=8g $MOUNTS \
  -e INSTANCE_LABEL=r6a.2xlarge --entrypoint python \
  s3d-bench:caiman /work/run_caiman_onacid.py --subset 2 --n-cores 8 \
    --memmap-path-file /results/memmap.txt
```

Note that if you have *already* run `--mode offline`, this registers a second
time: `run_caiman.py` never writes `memmap.txt`, and has no flag to reuse an
existing memmap. Expect the extra ~11 min (subset 2) / ~55 min (subset 10), and
see the `samples_*.npz` overwrite warning in §7.

Replace `--subset 2` with `--subset 10` for the longer run — **except for the
offline CNMF arm**, which does not fit in 64 GiB at that length (§7).

## 5. Parameters

All parameters live in `config.py` and are applied by every runner. Notably
`K = 1000` and `pw_rigid = True` for CaImAn; Suite3D runs 3D non-rigid
registration on the GPU and is given the 7 functional planes (2–8), while CaImAn
is given all 9 acquired planes.

There are no CLI flags for `K` or for `pw_rigid` — edit `config.py` to change
them. The published `N = 19` OnACID bars were produced with `K = 200` (and a
`rigid` variant), not with the `K = 1000` shipped here.

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
- Suite3D job directory under `scratch/`; CaImAn `results.hdf5` under `results/`
  (the offline one is valid; the OnACID one is truncated at `K = 1000` — see §7.7)

`INSTANCE_LABEL` (or `--instance`) only labels the `instance` column; it defaults
to `local`.

CPU percentages are core-equivalents: 800% means eight saturated vCPUs.

### Reading `timings.csv` correctly

- The `phase=total` row for **`caiman_onacid_p0` excludes registration** —
  `run_caiman_onacid.py` records `registration = 0.0` because it did not run it.
  The `suite3d` and `caiman_offline` totals *include* registration. To compare
  end-to-end wall times, add the `caiman_motion,registration` row to the OnACID
  total yourself.
- `avg_cpu_pct` / `avg_mem_gb` / `peak_mem_gb` are sampled **system-wide**
  (`psutil.cpu_percent`, `virtual_memory().used`), not from the container's
  cgroup, and `n_cpu_cores` is the **host** core count, not `--cpus`. These are
  only meaningful on a dedicated instance running nothing else. Run the benchmark
  on an otherwise idle box; the `local` default label is a smoke-test convenience.

## 7. Gotchas

Things that will bite you, roughly in the order you hit them.

1. **Wrong branch.** `benchmarking/paper/` is not on `main`. See §0.
2. **`.dockerignore`.** If `benchmarking/` is excluded without the
   `!benchmarking/paper/` negation, both builds fail at `COPY`. See §3.
3. **Storage.** Suite3D registration is I/O-bound; a default gp3 root volume
   makes it ~2–5× slower. See §2. **This is the single easiest way to publish a
   wrong number.**
4. **Offline CNMF does not fit in 64 GiB at `--subset 10`.** `detection_fit`
   peaks at 42.7 GB with `--subset 2` (T = 444 volumes); memory scales with T, and
   at `--subset 10` (T = 2220) a single float32 copy of `Y` is already ~21 GiB.
   At the full 19 files it was reported OOM-killed at every `K` from 50 to 1000
   and at both 1 and 8 workers. Only the OnACID and Suite3D arms scale to the
   long run on the instances above.
5. **Outputs are root-owned.** The containers run as root, so `results/` and
   `scratch/` fill with root-owned files. `sudo chown -R $USER results scratch`
   before you try to clean up or copy them.
6. **`samples_*.npz` collide.** `run_caiman.py --mode offline` and
   `run_caiman_register.py` both label their registration phase `caiman_motion`,
   so the second overwrites `samples_caiman_motion_registration_subset<N>.npz`
   and appends a *second* `caiman_motion,registration` row to `timings.csv`.
   Running the README in order does this. Move `results/` aside between arms if
   you care about the traces.
7. **OnACID exits 137 at the end, and loses two outputs.** With the shipped
   `K = 1000`, `cnm.save()` is OOM-killed after `fit_online()` returns (observed:
   63.3 GB RSS on the 64 GiB box). Consequences:
   - the `caiman_onacid_p0,detection_online` row and its `.npz` **are** written
     first, so the wall time is valid;
   - `write_total_row()` runs *after* `cnm.save()`, so there is **no
     `caiman_onacid_p0,total` row** in `timings.csv` — add registration +
     `detection_online` yourself;
   - `results.hdf5` is left truncated and unreadable (`OSError: bad object header
     version number`).

   The published `K = 200` runs did not hit this, which is why their CSVs *do*
   carry a `total` row.
8. **CaImAn is silent for the first ~10 min** (subset 2) while it rewrites the
   TIFFs to single-channel 3D stacks under `scratch/preprocessed_tifs/`. It is
   not hung. This is cached across runs.
9. **`--n-cores` help text says "locked to 16"**; it defaults to 16, but the
   instances above have 8 vCPU and every command here passes `--n-cores 8`.
10. **`--instance` defaults to `local`** and only labels the CSV column. Forget
    it and your rows are mislabelled.
11. **Frame rate.** `run_suite3d.py` sets `fs` from
    `suite3d.io.get_vol_rate()`, which reads ScanImage's
    `SI.hRoiManager.scanFrameRate` — the *plane* rate (~30 Hz), not the volume
    rate (~4.3 Hz for 9 planes). CaImAn is given `fr = 4` from `config.py`. This
    does not affect wall time (it only feeds deconvolution), but the two tools
    are not told the same frame rate.
12. **The suite3d image is ~15 GB.** `Dockerfile.suite3d` tries to install
    `psutil torch` from the PyTorch CPU index; `psutil` is not on that index, so
    the whole command fails and the `|| pip install psutil torch` fallback pulls
    the full CUDA torch from PyPI. (`pip install -e` has already pulled torch
    anyway.)
