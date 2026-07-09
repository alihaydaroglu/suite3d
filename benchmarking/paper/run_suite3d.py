"""Phase-resolved Suite3D runner for the speed comparison.

Two top-level phases, with detection broken into its natural sub-steps:

  registration       — run_init_pass + register
  detection_corrmap  — calculate_corr_map
  detection_segment  — segment_rois
  detection_npil     — compute_npil_masks
  detection_extract  — extract_and_deconvolve

Designed to run inside `Dockerfile.suite3d`. Raw TIFs bind-mounted at
/data, results at /results, scratch at /scratch.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import config  # noqa: E402
from timing_harness import PhaseTimer, write_total_row  # noqa: E402


def _cap_blas_threads(n: int) -> None:
    for env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(env, str(n))


def _gather_tifs() -> list[str]:
    from suite3d import io as s3d_io

    tifs: list[str] = []
    for d in config.RAW_TIF_DIRS:
        if not d.exists():
            raise SystemExit(f"Raw TIF dir missing: {d}")
        tifs += s3d_io.get_tif_paths(str(d))
    if not tifs:
        raise SystemExit(f"No tifs found under {config.RAW_TIF_DIRS}")
    return tifs


def _build_job(tifs: list[str], scratch_dir: Path, overwrite: bool,
               n_cores: int):
    from suite3d import io as s3d_io
    from suite3d.job import Job

    params = dict(config.S3D_JOB_PARAMS)
    params["fs"] = s3d_io.get_vol_rate(tifs[0])
    params["planes"] = np.asarray(config.FUNCTIONAL_PLANES)
    params["n_ch_tif"] = config.N_PLANES_TOTAL
    params["n_init_files"] = min(params.get("n_init_files", 4), len(tifs))

    job = Job(
        str(scratch_dir),
        config.S3D_JOB_ID,
        tifs=tifs,
        params=params,
        create=True,
        overwrite=overwrite,
        verbosity=3,
    )
    # Honour --n-cores for the parts of suite3d that respect it.
    job.params["n_proc_detect"] = n_cores
    job.save_params()
    return job


def main() -> None:
    ap = argparse.ArgumentParser(description="Suite3D speed-comparison runner")
    ap.add_argument("--dataset", default=config.DATASET)
    ap.add_argument(
        "--subset", type=int, default=None,
        help="If set, use only the first N TIF files (smoke test).",
    )
    ap.add_argument(
        "--scratch-dir", default="/scratch/suite3d",
        help="Where to create the Suite3D job (must be writable).",
    )
    ap.add_argument("--out-csv", default="/results/timings.csv")
    ap.add_argument(
        "--instance", default=os.environ.get("INSTANCE_LABEL", "local"),
    )
    ap.add_argument(
        "--n-cores", type=int, default=16,
        help="Worker count for detection. Locked to 16 to match the AWS plan.",
    )
    ap.add_argument(
        "--keep-scratch", action="store_true",
        help="Don't wipe scratch_dir before starting.",
    )
    args = ap.parse_args()

    _cap_blas_threads(1)

    tifs = _gather_tifs()
    if args.subset is not None:
        tifs = tifs[: args.subset]
    print(f"Suite3D speed-run on {args.dataset}: {len(tifs)} TIF(s) "
          f"(n_cores={args.n_cores})")

    scratch = Path(args.scratch_dir)
    if scratch.exists() and not args.keep_scratch:
        print(f"Wiping scratch_dir {scratch}")
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    # chdir AFTER rmtree+mkdir, otherwise CWD points at a deleted inode
    # and downstream os.getcwd() (e.g. numba error reporter) crashes.
    os.chdir(scratch)

    job = _build_job(tifs, scratch, overwrite=True, n_cores=args.n_cores)
    print(f"Job dir: {job.job_dir}")

    timer_kwargs = dict(
        out_csv=args.out_csv,
        dataset=args.dataset,
        instance=args.instance,
        subset=args.subset,
    )

    walls: dict[str, float] = {}

    # ---------- Registration ----------
    with PhaseTimer("suite3d", "registration", **timer_kwargs) as t:
        job.run_init_pass()
        job.register()
    walls["registration"] = t.wall_s

    # ---------- Detection (sub-phased) ----------
    for k, v in config.S3D_CORRMAP_PARAMS.items():
        job.params[k] = v
    for k, v in config.S3D_SEG_PARAMS.items():
        job.params[k] = v
    job.params["n_proc_detect"] = args.n_cores  # reapply after seg overrides
    job.save_params()

    seg_output = "seg"

    with PhaseTimer("suite3d", "detection_corrmap", **timer_kwargs) as t:
        job.calculate_corr_map()
    walls["detection_corrmap"] = t.wall_s

    with PhaseTimer("suite3d", "detection_segment", **timer_kwargs) as t:
        job.segment_rois(output_dir_name=seg_output)
    walls["detection_segment"] = t.wall_s

    stats_dir = os.path.join(job.job_dir, seg_output, "rois")

    with PhaseTimer("suite3d", "detection_npil", **timer_kwargs) as t:
        job.compute_npil_masks(stats_dir=stats_dir)
    walls["detection_npil"] = t.wall_s

    with PhaseTimer("suite3d", "detection_extract", **timer_kwargs) as t:
        job.extract_and_deconvolve(stats_dir=stats_dir, save_dir=stats_dir)
    walls["detection_extract"] = t.wall_s

    write_total_row(
        args.out_csv, tool="suite3d",
        dataset=args.dataset, instance=args.instance, subset=args.subset,
        walls_s=walls,
    )

    print(f"\nDone. Timing rows appended to {args.out_csv}")


if __name__ == "__main__":
    main()
