# Suite3D demos

Four self-contained demos. Each one takes a downloaded dataset, runs the full
Suite3D pipeline on it, and opens a viewer.

| | demo | recording | what it shows |
|---|---|---|---|
| **01** | [`01-v1-tc030/`](01-v1-tc030/) | V1, standard 2P, 7 planes | The baseline. **No parameter overrides at all** beyond acquisition geometry — this is Suite3D out of the box. Start here. |
| **02** | [`02-lbm-ss004/`](02-lbm-ss004/) | LBM, 22 planes, ~40k neurons | Light-beads microscopy: strip fusion, cavity crosstalk, plane reordering. The large one. |
| **03** | [`03-hippocampus/`](03-hippocampus/) | CA1, standard 2P, 4 planes | Densely packed somata, and a shallow stack that needs `apply_z_shift=False`. |
| **04** | [`04-sweep/`](04-sweep/) | (reuses demo 03) | How detection parameters were chosen: two 3×3 sweeps, opened in the napari sweep viewer. |

Each dataset demo has a `run_pipeline.py` (one command, start to finish) and a
`walkthrough.ipynb` (the same pipeline stage by stage, plotting as it goes).
Shared helpers live in [`common/`](common/).

## Getting the data

The demos **do not download anything**. Get the example datasets from the
figshare upload and point `--data-root` at the folder you unpacked:

```
<data-root>/
    v1/raw/*.tif
    hippocampus/raw/*.tif
    lbm/raw/*.tif
    manifest.json
```

`--data-root` also accepts a single dataset folder (`.../v1`) or its `raw/`
directory, if you only downloaded one.

## Setup

Suite3D is **not on PyPI** — install it from git:

```bash
pip install git+https://github.com/alihaydaroglu/suite3d.git
pip install 'cupy-cuda12x>=13.0,<14.0'    # GPU; see below
```

A GPU is used for registration (`gpu_reg=True`). Everything has a CPU fallback,
but registration is much slower without one.

> `pip install suite3d[gpu]` installs **nothing** — the `gpu` extra in
> `pyproject.toml` is empty. Install cupy yourself, and pin it to the 13.x
> line: cupy 14.x bundles NVRTC 13.0, whose kernels CUDA-12.x drivers reject
> with `CUDA_ERROR_INVALID_IMAGE`.

For `--viewer napari`, also `pip install 'suite3d[viz]'`.

## Running on AWS

See [`aws/README.md`](aws/README.md) — instance choice, disk sizing, a
`bootstrap.sh` provisioner, a Dockerfile, and the measured cost of demo 01
(25.4 min, 22 GB peak on a `g4dn.2xlarge`).

## Running

```bash
cd 01-v1-tc030
python run_pipeline.py --data-root /path/to/figshare --out-dir ./results
```

This writes a job directory at `./results/s3d-demo-v1/` and, at the end,
exports the results and opens a viewer.

```bash
--viewer html      # portable offline browser (default)
--viewer napari    # the desktop 3D viewer
--viewer none
```

Stages can be skipped individually to re-use what is already on disk — handy
for re-opening a viewer, or re-running only segmentation:

```bash
python run_pipeline.py --out-dir ./results \
    --skip-init --skip-register --skip-corrmap --viewer napari
```

Use `--n-frames N` to clip the correlation map to the first `N` volumes.
Registration always reads whole tifs.

> The `--viewer html` path calls `job.make_html_viewer()`. If your build of
> suite3d predates it, the run still exports its results and tells you so; use
> `--viewer napari` in the meantime.

## Cost

Plan for RAM and disk, not just time. Measured end to end on one box (RTX A4500,
8 cores, raw data on a local disk). Registration is usually bound by how fast you
can read the raw tifs — the same demo over a network mount took several times
longer.

| demo | download | registered movie | peak RAM | wall time (GPU) |
|---|---|---|---|---|
| 01 V1 | 21.1 GB (10 tifs) | ~9 GB | 13.6 GiB | **4m51s** (reg 2m19s) |
| 02 LBM | 56.2 GB (13 tifs) | ~42 GB | **113.5 GiB** | **39m45s** (reg 10m36s) |
| 03 hippocampus | 21.1 GB (10 tifs) | ~9 GB | 8.1 GiB | **5m24s** (reg 2m12s) |

> ⚠ **Demo 02 peaks at ~114 GiB of RAM**, and it peaks during *trace extraction*,
> not registration. It will not run on a 64 GB machine. Demos 01 and 03 fit
> comfortably in 16 GB.

The results directory written by `export_results(..., make_viewer=True)` is
~360 MB for demo 01 and ~3.7 GB for demo 02.

Demo 04 re-uses demo 03's job. Its corrmap sweep is cheap (9 correlation maps,
no re-registration); its segmentation sweep is not.

Expect roughly 885 ROIs on demo 01, 1,350 on demo 03, and ~40,000 on demo 02.
Treat these as sanity checks, not regression targets.

**On one machine the count is reproducible; across machines it drifts.** Demo 01
gives 885 ROIs on every re-run here, and 878 on an AWS T4 — a 0.8% difference
that comes from GPU and BLAS arithmetic, not from randomness in the algorithm.
Detection extends each ROI by power iteration, and although the residual source
does start from an unseeded random vector, power iteration converges to the same
dominant component regardless of where it starts. If you need bit-identical
counts across hardware, you cannot have them; if you re-run on the same box and
the count moves, something else changed.

Do not compare against the 845 ROIs quoted by demo 01's own published reference
run. That run's `params.npy` was overwritten after the fact, so its recorded
parameters are not the ones it used, and it segmented a `(7, 514, 514)` volume
where the demo segments `(7, 515, 514)`. The two numbers are not measuring the
same thing.

Demo 02's detection and segmentation parameters are pinned to those that produced
the published 43,652-ROI segmentation of that recording. Its registration is not:
the demo uses nonrigid, the published run was rigid-only. Registration feeds the
correlation map, so its count will sit near 43,652 for two independent reasons.

## Things that are easy to get wrong

**`fs` is the volume rate, not the plane rate.** Suite3D derives
`detection_timebin = 2 * round(fs / tau)` from it, so a wrong `fs` changes the
*correlation map*, not merely a plot's time axis. `suite3d.io.get_vol_rate()`
returns the **per-plane** rate despite its name — the demos never call it, they
hard-code `fs` per dataset in [`common/datasets.py`](common/datasets.py) and
assert the value is plausible. Use `SI.hRoiManager.scanVolumeRate`.

**`voxel_size_um` is `(dz, dy, dx)` and is not cosmetic.** The correlation map
divides `cell_filt_z_um` and `npil_filt_z_um` by `dz`.

**`apply_z_shift=False` is not the same as `3d_reg=False`.** On shallow stacks
(≲4 planes) the axial phase correlation saturates at the edge of its search
window, so demo 03 measures z in 3D and simply does not apply it. `3d_reg` stays
on in all three demos, and should stay on in yours.

**`multi_source` shipped as `True` until 2026-07-09 and now defaults to
`False`.** The three reference runs used it, so the demos set it explicitly.

**Sweeps rewrite `params.npy`.** `Job.save_params()` writes the job's root
parameter file on every mutation, and a sweep mutates many times.
`04-sweep/run_sweep.py` snapshots and restores it in a `finally:` block. Do the
same if you write your own.

**Do not set `extend_thresh` or `min_frames`.** Both are accepted and both do
nothing: neither reaches the segmentation code. The wired equivalent of
`extend_thresh` is `vox_snr_thresh`.
</content>
