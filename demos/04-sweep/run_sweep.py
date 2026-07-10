#!/usr/bin/env python
"""Suite3D demo 04 — parameter sweeps on the hippocampus recording.

Two 3x3 sweeps, run on the dataset from demo 03:

* **corrmap sweep** — `cell_filt_xy_um` x `intensity_thresh`. These set,
  respectively, the lateral scale Suite3D looks for and how much of the movie
  survives the per-voxel noise gate. Together they largely determine the
  *diameter* of what gets detected.
* **segmentation sweep** — `segmentation_spatial_filt` x `vox_snr_thresh`.
  The first smooths each frame in xy before peak detection; the second is the
  fraction of a voxel's variance the ROI's trace must explain for that voxel to
  join the ROI. Raising it yields **smaller** ROIs, and therefore **more** of
  them, as footprints shrink and split. Measured on demo 03's job, at
  `segmentation_spatial_filt=2`, for `vox_snr_thresh` 0.05 / 0.10 / 0.20:
  1224 / 1393 / 1601 ROIs, of median 556 / 390 / 265 voxels. Smoothing harder
  (`segmentation_spatial_filt` up) does the opposite: fewer, larger ROIs.

`Job.setup_sweep` requires the job's *current* value of each swept parameter to
appear in that parameter's list, so every list below brackets the default.

Two things to know before running this on your own data
-------------------------------------------------------
1. The corrmap sweep re-runs the correlation map for each of its 9 cells, which
   is cheap. The segmentation sweep re-runs detection, which is not. Registration
   is not repeated by either.
2. `Job.save_params()` rewrites the job's root `params.npy` on every mutation,
   and a sweep mutates params many times. This script snapshots `params.npy`
   before it starts and restores it in a `finally:` block, so an interrupted
   sweep cannot leave your job pinned to the last cell's parameters. If you
   write your own sweep, do the same.

Usage
-----
    # run against the job produced by ../03-hippocampus/run_pipeline.py
    python run_sweep.py --job-dir ./results/s3d-demo-hippocampus

    python run_sweep.py --job-dir ... --which corrmap
    python run_sweep.py --job-dir ... --which seg

Then open the result:

    python open_sweep_napari.py --sweep-dir ./results/s3d-demo-hippocampus/sweeps/corrmap
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import log   # noqa: E402
from suite3d.job import Job   # noqa: E402


# Each list brackets the hippocampus job's current value (setup_sweep requires it).
CORRMAP_SWEEP = {
    "cell_filt_xy_um": [0.5, 1.5, 2.5],   # default 1.5
    "intensity_thresh": [1, 3, 5],        # default 3
}

SEG_SWEEP = {
    "segmentation_spatial_filt": [0, 2, 4],   # default 2
    "vox_snr_thresh": [0.05, 0.10, 0.20],     # default 0.10
}


def load_job(job_dir):
    job_dir = Path(job_dir).expanduser().resolve()
    if not (job_dir / "params.npy").exists():
        raise FileNotFoundError(
            "No params.npy in %s — point --job-dir at a finished suite3d job "
            "directory (run ../03-hippocampus/run_pipeline.py first)." % job_dir
        )
    # Job(root, job_id) where job_dir == root/s3d-<job_id>
    job_id = job_dir.name[len("s3d-"):] if job_dir.name.startswith("s3d-") else job_dir.name
    return Job(str(job_dir.parent), job_id, create=False), job_dir


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job-dir", required=True,
                   help="finished hippocampus job dir (…/s3d-demo-hippocampus)")
    p.add_argument("--which", choices=["corrmap", "seg", "both"], default="both")
    args = p.parse_args()

    job, job_dir = load_job(args.job_dir)

    # See docstring note 2: a sweep mutates the root params.npy.
    root = job_dir / "params.npy"
    backup = job_dir / "params.npy.sweep_backup"
    if not backup.exists():
        shutil.copy2(root, backup)
        log("Snapshotted %s" % root.name)

    try:
        if args.which in ("corrmap", "both"):
            log("=== corrmap sweep: %s ===" % CORRMAP_SWEEP)
            job.sweep_corrmap(CORRMAP_SWEEP, sweep_name="corrmap")

        if args.which in ("seg", "both"):
            log("=== segmentation sweep: %s ===" % SEG_SWEEP)
            job.sweep_segmentation(SEG_SWEEP, sweep_name="seg", all_combinations=True)
    finally:
        shutil.copy2(backup, root)
        log("Restored the job's root params.npy")

    sweeps = job_dir / "sweeps"
    log("Sweeps written under %s" % sweeps)
    log("Open one with:  python open_sweep_napari.py --sweep-dir %s"
        % (sweeps / ("corrmap" if args.which != "seg" else "seg")))


if __name__ == "__main__":
    main()
