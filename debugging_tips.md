# Suite3D debugging tips (for AI agents)

A growing list of known problems, their symptoms, and what to check.
Add new entries when you diagnose something non-obvious. Keep entries
short, lead with the symptom (that's how a future agent will search),
and include file:line references where the failure mode lives.

---

## Saturating z-shifts → step-wise zeroed traces

**Symptoms**

- Extracted F traces drop to ≈ 0 (or strongly negative after Fneu
  subtraction) for a stretch of consecutive frames, then recover.
  Often population-wide, not per-cell.
- A burst of frames where `sub_pixel_shifts[:, 0]` (z) sits at
  exactly `±(pc_size_z + 0.5)` — the literal corner of the rigid
  search window. With default `pc_size = (2, 40, 40)` that's
  `±2.5`.
- The post-registration log line *"Saturation diagnostic: N/M
  frames (P%) hit the z-axis search-window edge ±X.X"* — emitted by
  [`_log_rigid_saturation_diagnostic`](suite3d/iter_step.py) when any
  axis hits the edge.

**Cause**

Phase-correlation peak failure, not real drift. When the PC volume
has no clean maximum (low SNR, illumination dropout, sample
disturbance, or simply too few z planes to support a real 3D
correlation), `est_sub_pixel_shift`
([reg_3d.py:172](suite3d/reg_3d.py)) falls back to the corner of the
search window via its periodic-wrap term. The "saturation" is the
estimator giving up, not real ±N-plane drift.

Validation: PC peak heights at saturated frames are typically much
lower than at unsaturated frames (often 50%+ lower median). See the
ATL020 diagnostic workup at
`/mnt/md0/s3d-revisions/figures/atl020-saturation-diag/review.html`
for the full pattern.

**Notation gotcha** (related, easy to misread the offsets dict):
`sub_pixel_shifts` is **not** the sub-pixel offset alone. Despite the
name, `est_sub_pixel_shift` returns `shift + 0.5*sub_pixel` where
`shift` is the integer-from-center, so `sub_pixel_shifts` is the
*full* shift (integer + half-parabolic). The applied shift in
[reg_3d.py:1158-1168](suite3d/reg_3d.py) uses `sub_pixel_shifts`
directly and never re-adds `int_shift`. Do not compute
`int_shift + sub_pixel_shifts` — it double-counts the integer.
A frame at the negative-z corner reads
`int_shift_z = -pc_size_z` AND `sub_pixel_shifts_z = -(pc_size_z+0.5)`
simultaneously; the actual applied shift is just the second value.

**Fixes**

- **First check `pc_size_z`.** It's the maximum integer shift
  searched in z. The hard upper bound is `nz_planes / 2`, since the
  3D phase-correlation output only has as many z slices as the
  recording. With 4 imaged planes you cannot widen beyond
  `pc_size_z = 2`. Trying anyway crashes in `process_phase_corr_gpu`
  (shape-mismatch on the centered-window slicing).
- **If you have plenty of z planes** and saturation is occurring at
  the existing window, widening `pc_size = (5, 40, 40)` is reasonable
  to test — but verify the wider window relaxes the saturated frames
  to sensible values rather than just relocating the corner.
- **If you have few z planes** (≤4), or saturation persists at the
  hardware limit, **disable z correction**: `3d_reg=False`. The 2D
  path runs per-plane y/x rigid registration without z, which avoids
  the failure mode entirely. Cell footprints and downstream
  extraction are unaffected by skipping z; only the F-trace zeroing
  artifact goes away.
- Do **not** "fix" by clamping or remapping `sub_pixel_shifts` after
  the fact. The right answer is either to make the algorithm
  estimate well (more planes, better SNR) or to bypass z entirely.

**Reference incident**

ATL020_2023-04-12-GRN: 440/6708 frames pegged at z=−2.5 in the
688–1023 s window, zeroing F traces population-wide. 4 z-planes,
default `pc_size_z = 2`. PC peak heights at saturated frames had
median 0.0148 vs 0.0370 unsat. Fix: re-register with `3d_reg=False`.
Memo: [dev/coordination/rebuttal.md](../dev/coordination/rebuttal.md)
"ATL020: disable z-axis registration".

---

## Oversized TIFFs → OOM during load

**Symptoms**

- Job crashes (or the process is killed by the OS) during the load
  step of `run_init_pass()` or `register()`, before any registration
  log lines appear.
- `MemoryError` from inside
  [`_load_scanimage_tifs`](suite3d/io/s3dio.py) at the
  `tifffile.imread(tif_path)` call, or system OOM-killer fires.
- `preregister_tifs` prints a level-0 warning at job creation:
  *"Detected unusually large TIFF file(s). These may exhaust RAM
  during the load step..."*. Default thresholds: > 1000 frames OR
  > 10 GB per file.

**Cause**

The ScanImage 2P loader reads each tif into RAM in a single
`tifffile.imread(path)` call (see [s3dio.py:169](suite3d/io/s3dio.py)).
The pipeline's unit of work is one whole tif: it must fit in memory.
A 20 000-frame × 30-plane × 512² × int16 tif is ≈ 600 GB; even a
5000-frame tif at a large FOV easily exceeds 100 GB. LBM/FACED data
go through different paths and aren't affected by this specific
failure mode (their crashes look different).

**Fix**

Pre-split the offending tifs into volume-aligned chunks using the
shipped CLI utility. The split is page-granular and streams through
the source via `tifffile.memmap`, so it works on files larger than
RAM:

    s3d-split-tiff <tif_path> <output_dir> \
        --frames-per-chunk 500 \
        --n-ch-tif <planes_per_volume> \
        --num-colors <num_colors>

`--frames-per-chunk` is rounded down to a multiple of
`n_ch_tif * num_colors` so each chunk holds whole volumes; only the
final chunk may have a partial trailing volume, which the existing
extra-frames spillover logic in s3dio handles transparently. Output
files are named `<src_basename>_<NNNN>.tif`.

Then point your `Job` at the split tifs instead of the original.
Suite3D sees N normal small tifs and runs unchanged.

**Cost**

I/O-bound: read + write of the entire source. On NVMe RAID
(`/mnt/md0`), a 600 GB source splits in ~5-15 min. On slower
storage, longer. This is a one-time cost per problematic dataset;
subsequent suite3d runs use the split files directly.

**Why no in-pipeline fix**

Avoiding the read+write would require refactoring suite3d's loader
and `iter_step.py` batching to operate on sub-file segments rather
than whole tifs (~500-700 LOC across 4-5 files, plus a regression
test matrix). For an edge case affecting one experiment, the
splitting utility's wall-clock cost is cheaper than the engineering
cost.

**Reference**

CLI is registered in `pyproject.toml` as the `s3d-split-tiff`
entry point; implementation in
[tiff_utils.py:split_oversized_tiff](suite3d/io/tiff_utils.py).
Preregister warning at
[job.py:_warn_if_oversized_tifs](suite3d/job.py).
