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

> ⚠ **Read [§7 Gotchas](#7-gotchas) before you start.** Three of them — where
> `scratch/` lives (§7.3), the offline-CNMF memory ceiling (§7.4), and the fact
> that OnACID motion-corrects a second time inside its *detection* phase (§7.8) —
> will silently give you a wrong number or an OOM kill if you skip them.

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

`--mode` defaults to `offline`. `--mode both` also runs OnACID (with `p = 0`,
see §5) on the same registration, in one container. The two-step recipe below
is equivalent and lets you re-run OnACID without re-registering.

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

`run_caiman.py` also writes `results/memmap.txt` after its registration, so if
you have already run `--mode offline` you can skip `run_caiman_register.py` and
go straight to `run_caiman_onacid.py` against the same memmap.

Replace `--subset 2` with `--subset 10` for the longer run. Reference wall
times at `--subset 10`, 8 vCPU, with the **round-1 parameters**: Suite3D 635 s
total (on NVMe); CaImAn NoRMCorre 3417 s; OnACID `detection_online` 2126 s; the
offline CNMF arm was OOM-killed in 64 GiB at that length (§7.4). With the
corrected parameters (§8) the offline arm is far lighter — see the table there.

## 5. Parameters

All parameters live in `config.py` and are applied by every runner. CaImAn
runs with `config.CAIMAN_PARAMS`, the **corrected round-2 set** (§8): `p = 1`,
`gSig = (1, 5, 5)`, patches with `K = 22` per patch (`rf = 40`, `stride = 18`,
`only_init_patch = True`), piecewise-rigid NoRMCorre with
`strides = (3, 96, 96)`, `overlaps = (1, 32, 32)`, `max_shifts = (1, 5, 5)`.
All per-axis tuples are in the data's `(z, y, x)` order. Suite3D runs 3D
non-rigid registration on the GPU and is given the 7 functional planes (2–8),
while CaImAn is given all 9 acquired planes.

`--legacy-params` on any CaImAn runner selects `config.CAIMAN_PARAMS_LEGACY`,
the set shipped with the round-1 benchmark (`K = 1000` over the whole FOV,
`gSig = (5, 5, 2)`, `p = 2`, motion tuples in `(x, y, z)` order, and OnACID
re-running motion correction). It exists so the correction can be reproduced;
do not use it for new measurements. There are no CLI flags for individual
parameters — edit `config.py`. The published `N = 19` OnACID bars were produced
with `K = 200` (and a `rigid` variant) of the legacy set.

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
4. **Offline CNMF does not fit in 64 GiB at `--subset 10`.** Verified: `CNMF.fit`
   on the `--subset 10` memmap (`T = 2220`, one float32 copy of `Y` = 19.5 GiB) is
   OOM-killed — `anon-rss 62.6 GB`, exit 137. For reference `detection_fit` peaks
   at 42.7 GB with `--subset 2` (`T = 444`), and memory scales with T. At the full
   19 files it was reported OOM-killed at every `K` from 50 to 1000 and at both 1
   and 8 workers. Only the OnACID and Suite3D arms scale to the long run on the
   instances above.
5. **Outputs are root-owned.** The containers run as root, so `results/` and
   `scratch/` fill with root-owned files. `sudo chown -R $USER results scratch`
   before you try to clean up or copy them.
6. **`samples_*.npz` collide.** `run_caiman.py --mode offline` and
   `run_caiman_register.py` both label their registration phase `caiman_motion`,
   so the second overwrites `samples_caiman_motion_registration_subset<N>.npz`
   and appends a *second* `caiman_motion,registration` row to `timings.csv`.
   Running the README in order does this. Move `results/` aside between arms if
   you care about the traces.
7. **OnACID exits 137 at the end, and loses two outputs — with the legacy
   `K = 1000`.** `cnm.save()` is OOM-killed after `fit_online()` returns
   (observed: 63.3 GB RSS on the 64 GiB box). Not seen with the corrected
   parameters. Consequences, if you run `--legacy-params`:
   - the `caiman_onacid_p0,detection_online` row and its `.npz` **are** written
     first, so the wall time is valid;
   - `write_total_row()` runs *after* `cnm.save()`, so there is **no
     `caiman_onacid_p0,total` row** in `timings.csv` — add registration +
     `detection_online` yourself;
   - `results.hdf5` is left truncated and unreadable (`OSError: bad object header
     version number`).

   The published `K = 200` runs did not hit this, which is why their CSVs *do*
   carry a `total` row.
8. **OnACID used to motion-correct a second time, inside `detection_online`.**
   The round-1 runner did not set `online.motion_correct`, and CaImAn's default
   is `True`, so `fit_online()` re-registered a memmap that NoRMCorre had already
   registered (a full `MotionCorrect` on the `init_batch` frames, then a
   per-frame 3D rigid `register_translation_3d` on every streamed frame — visible
   on disk as `tmp_mov_mot_corr_rig__..._frames_200.mmap`). Measured on
   `r6a.2xlarge`, subset 2: 564.8 s with it, **439.8 s** without, i.e. 22 % of
   the phase was redundant MC. The corrected parameter set sets
   `motion_correct = False`; `--legacy-params` keeps the old behaviour.

9. **CaImAn is silent for the first ~10 min** (subset 2) while it rewrites the
   TIFFs to single-channel 3D stacks under `scratch/preprocessed_tifs/`. It is
   not hung. This is cached across runs.
10. **`--n-cores` defaults to 8** for the CaImAn runners (the instances above
   have 8 vCPU) and to 16 for `run_suite3d.py`; every command here passes
   `--n-cores 8` explicitly.
11. **`--instance` defaults to `local`** and only labels the CSV column. Forget
    it and your rows are mislabelled.
12. **Frame rate.** `run_suite3d.py` sets `fs` from
    `suite3d.io.get_vol_rate()`, which reads ScanImage's
    `SI.hRoiManager.scanFrameRate` — the *plane* rate (~30 Hz), not the volume
    rate (~4.3 Hz for 9 planes). CaImAn is given `fr = 4` from `config.py`. This
    does not affect wall time (it only feeds deconvolution), but the two tools
    are not told the same frame rate.
13. **The suite3d image is ~15 GB.** `Dockerfile.suite3d` tries to install
    `psutil torch` from the PyTorch CPU index; `psutil` is not on that index, so
    the whole command fails and the `|| pip install psutil torch` fallback pulls
    the full CUDA torch from PyPI. (`pip install -e` has already pulled torch
    anyway.)

## 8. Round-2 corrections to the CaImAn arm

In revision round 2 a reviewer re-ran these scripts and found the CaImAn
configuration wrong in several concrete ways. Their corrections are adopted
verbatim in `config.CAIMAN_PARAMS` and the runners; the round-1 configuration
is kept as `CAIMAN_PARAMS_LEGACY` (`--legacy-params`) so the before/after can be
reproduced. What was wrong:

1. **Axis order.** CaImAn's demo assumes `(t, x, y, z)`; this data is
   `(t, z, y, x)`. Every per-axis tuple was therefore transposed:
   `gSig = (5, 5, 2)` → `(1, 5, 5)`, `se = (3, 3, 1)` → `(1, 3, 3)`,
   `strides = (24, 24, 10)` → `(3, 96, 96)`, `overlaps = (12, 12, 2)` →
   `(1, 32, 32)`, `max_shifts = (4, 4, 2)` → `(1, 5, 5)`. The motion-correction
   patch (strides + overlaps) is now 4 × 128 × 128, the size of Suite3D's
   non-rigid blocks.
2. **No patches.** `K = 1000` components were sought over the whole FOV in a
   single process. Now `rf = 40`, `stride = 18`, `K = 22` per patch,
   `only_init_patch = True` (the `CNMFParams` default): initialisation runs in
   1 × 8 × 8 patches of 9 × 80 × 80 voxels in parallel, refinement in `refit`.
   `p = 1` instead of 2 at a 4 Hz volume rate.
3. **Multi-core paths not used.** `dview` was not passed to `refit()`, so the
   refit ran single-process. And because `fnames` was given to the
   motion-correction `CNMFParams`, `change_params()` reset `splits_rig/els` to
   `T // 80 = 2`, so NoRMCorre ran on 2 processes whatever `--n-cores` said.
   `fnames` is now omitted there and the splits are set to the worker count.
4. **(From round 1's own audit)** OnACID re-ran motion correction on registered
   input (§7.8); now off. `run_caiman.py` writes `memmap.txt` so the OnACID
   arm reuses the registration instead of paying for it twice.

Two things the corrections do **not** change, and that the OnACID numbers
must be read with:

- **3D OnACID returns exactly `K` components.** In every run we have made
  (K = 22, 200, 500, 1000; subsets 2, 10 and the full recording) the online
  phase never added a component, so `K` sets the size of the model being
  timed rather than a ceiling on detection. The reviewer's corrected
  `K = 22` therefore gives an OnACID run that "detects" 22 cells on a field
  with several hundred. `run_caiman_onacid.py --K` overrides it; the table
  below reports both K = 22 (the reviewer's configuration as run) and
  K = 500 (a model the size of Suite3D's cell count).
- **The patch parameters apply to offline CNMF only.** Passing `rf` / `stride`
  into the OnACID initialisation seeds the online phase with ~950 components
  and was 8× slower (3053 s vs 372 s at subset 2); the OnACID dict does not
  carry them, matching the script the reviewer timed.

### Before / after, measured on one machine

All rows below were run on the same box (72-core Xeon w9-3475, 503 GB, RTX
A4500) inside the `s3d-bench:caiman` image (caiman 1.12.2) with
`--cpus=8 --n-cores 8`, so the two CaImAn columns differ only in the scripts:
"shipped" is the round-1 tree at commit `fb87bef`, "corrected" is this one.
Wall seconds; peak RAM is system-wide. The reviewer's own numbers (10-core
W-2255, `--n-cores 20`) are given for reference.

**`--subset 2` (444 volumes, 9 × 512 × 512)**

| phase | shipped | corrected | speed-up | peak RAM shipped → corrected | reviewer: shipped → fixed |
|---|---:|---:|---:|---|---|
| NoRMCorre registration (pw-rigid) | 600 | **171** | 3.5× | 50 → 27 GB | 852 → 293 |
| CNMF fit | 893 | **120** | 7.5× | 57 → 24 GB | 922 → 78 |
| evaluate_components | 3.3 | 3.0 | — | | 7.2 → 6.9 |
| CNMF refit | 2,642 | **527** | 5.0× | 24 → 29 GB | 3,213 → 383 |
| **offline total** | **4,138** | **821** | **5.0×** | | 4,994 → 761 |
| OnACID `detection_online`, shipped K = 1000 (+ double MC) → corrected K = 22 | 1,037 | **165** | 6.3× | 53 → 32 GB | 1,061 → 372 |
| OnACID `detection_online`, corrected K = 500 | 1,037 | **413** | 2.5× | 53 → 39 GB | |
| Suite3D, same box, GPU (164 ROIs) | | **242** | | | 134 (reviewer's box) |

**`--subset 10` (2,220 volumes)**

| phase | shipped | corrected | speed-up | peak RAM shipped → corrected |
|---|---:|---:|---:|---|
| NoRMCorre registration (pw-rigid) | 4,455 | **833** | 5.3× | 26 → 28 GB |
| CNMF fit | OOM at 63 GB (AWS, round 1) | **165** | | → 40 GB |
| evaluate_components | | 7.7 | | |
| CNMF refit | | **623** | | → 45 GB |
| **offline total** | did not finish | **1,629** | | |
| OnACID `detection_online`, shipped K = 1000 (+ double MC) → corrected K = 22 | 4,807 | **890** | 5.4× | 52 → 32 GB |
| OnACID `detection_online`, corrected K = 500 | 4,807 | **1,818** | 2.6× | 52 → 39 GB |
| Suite3D, same box, GPU (408 ROIs) | | **301** | | |

Component counts with the corrected parameters: offline CNMF 378 (subset 2)
and 558 (subset 10) after refit; OnACID exactly K. Suite3D found 164 and 408
ROIs.

**What this means for the comparison.** Against the corrected CaImAn on this
machine, Suite3D end-to-end is 3.4× (subset 2) and 5.4× (subset 10) faster
than offline CNMF, and 2.4× / 8.8× faster than NoRMCorre + OnACID with a
500-component model (1.4× / 5.7× with the reviewer's 22-component model). The
round-1 configuration overstated CaImAn's offline time by 5× and its
registration by 3.5–5.3×; the corrected offline CNMF also fits comfortably in
64 GB at subset 10, where the shipped configuration was OOM-killed. The
published `N = 19` numbers were measured on AWS with the shipped
configuration and have not been re-measured there.

