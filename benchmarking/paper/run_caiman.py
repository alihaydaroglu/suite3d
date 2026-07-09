"""CaImAn speed-comparison runner — both online (OnACID) and offline
(CNMF.fit) detection paths, on a single motion-correction pass.

The reviewer ran approach-1 of `demo_realtime_cnmfE.ipynb` (online
CNMF, CNMF-E swapped for CNMF, 3D, multi-frame parallel). We mirror
that AND also report offline CNMF.fit() on the same data and params,
so the comparison covers both common CaImAn usage modes.

Phases timed (per run):
  registration       — MotionCorrect (NoRMCorre); run once.
  detection_onacid   — OnACID.fit_online (init batch + online refit).
  detection_offline  — offline CNMF.fit + estimates.evaluate_components.

Each "tool" string in the CSV is namespaced so the figures can split:
  caiman_onacid, caiman_offline.

Both share the same registration wall time (same motion-corrected
memmap), which is appended under each tool to keep totals coherent.
"""
from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
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


def _ensure_preprocessed_tifs(prep_dir: Path, subset: int | None) -> list[str]:
    """CaImAn expects single-channel 3D TIFs in (T, nz, ny, nx)."""
    import tifffile

    raw_tifs: list[str] = []
    for d in config.RAW_TIF_DIRS:
        d = Path(d)
        if not d.exists():
            raise SystemExit(f"Raw TIF dir missing: {d}")
        raw_tifs += sorted(str(p) for p in d.glob("*.tif"))
    if not raw_tifs:
        raise SystemExit(f"No raw tifs under {config.RAW_TIF_DIRS}")
    if subset is not None:
        raw_tifs = raw_tifs[:subset]

    prep_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for path in raw_tifs:
        name = Path(path).stem + f"_ch{config.FUNCTIONAL_COLOR_CHANNEL}.tif"
        out_path = prep_dir / name
        if out_path.exists():
            out.append(str(out_path))
            continue
        data = tifffile.imread(path)
        if data.ndim == 4:
            green = data[:, config.FUNCTIONAL_COLOR_CHANNEL]
        elif data.ndim == 3:
            green = data
        else:
            raise ValueError(f"Unexpected TIF shape {data.shape} ({path})")
        del data
        nz = config.N_PLANES_TOTAL
        nvol = green.shape[0] // nz
        green = green[: nvol * nz].reshape(nvol, nz, green.shape[1], green.shape[2])
        tifffile.imwrite(out_path, green.astype(np.int16))
        del green
        gc.collect()
        out.append(str(out_path))
    return out


def _run_registration(tifs, dview, timer_kwargs, tool_label):
    """Time NoRMCorre under the given tool label. Returns (memmap, wall_s)."""
    import caiman as cm
    from caiman.motion_correction import MotionCorrect
    from caiman.source_extraction.cnmf import params as cnmf_params

    mc_opts = cnmf_params.CNMFParams(params_dict={
        "fnames": tifs,
        "strides": config.CAIMAN_PARAMS["strides"],
        "overlaps": config.CAIMAN_PARAMS["overlaps"],
        "max_shifts": config.CAIMAN_PARAMS["max_shifts"],
        "max_deviation_rigid": config.CAIMAN_PARAMS["max_deviation_rigid"],
        "pw_rigid": config.CAIMAN_PARAMS["pw_rigid"],
        "is3D": True,
    })
    mc = MotionCorrect(tifs, dview=dview, **mc_opts.get_group("motion"))

    with PhaseTimer(tool_label, "registration", **timer_kwargs) as t:
        mc.motion_correct(save_movie=True)

    mc_fnames = (
        getattr(mc, "fname_tot_els", None)
        or getattr(mc, "fname_tot_rig", None)
        or mc.mmap_file
    )
    fname_new = cm.save_memmap(mc_fnames, base_name="memmap_", order="C")
    return fname_new, t.wall_s


def _run_onacid(memmap_path, dview, n_proc, timer_kwargs, results_dir):
    """OnACID online CNMF — mirrors `demo_realtime_cnmfE.ipynb`. Returns wall_s."""
    from caiman.source_extraction.cnmf import params as cnmf_params
    from caiman.source_extraction.cnmf.online_cnmf import OnACID

    p = config.CAIMAN_PARAMS
    opacid_dict = {
        "fnames": [memmap_path],
        "fr": p["fr"], "decay_time": p["decay_time"],
        "gSig": p["gSig"], "p": p["p"], "K": p["K"],
        "merge_thr": p["merge_thresh"], "rval_thr": p["rval_thr"],
        "min_SNR": p["min_SNR"], "use_cnn": p["use_cnn"],
        "is3D": True,
        "init_method": p["init_method"],
        "init_batch": p["init_batch"],
        "epochs": p["epochs"], "n_refit": p["n_refit"],
    }
    opts = cnmf_params.CNMFParams(params_dict=opacid_dict)
    cnm = OnACID(params=opts, dview=dview)

    with PhaseTimer("caiman_onacid", "detection", **timer_kwargs) as t:
        cnm.fit_online()

    results_dir.mkdir(parents=True, exist_ok=True)
    cnm.save(str(results_dir / "results.hdf5"))
    return t.wall_s


def _run_offline(memmap_path, dview, n_proc, timer_kwargs, results_dir):
    """Offline CNMF — matches the existing fig-comparison-contained
    pipeline (step2_caiman.py). Returns dict of sub-phase wall times."""
    import caiman as cm
    from caiman.source_extraction.cnmf.cnmf import CNMF

    p = config.CAIMAN_PARAMS
    Yr, dims, T = cm.load_memmap(memmap_path)
    images = np.reshape(Yr.T, [T] + list(dims), order="F")

    cnm = CNMF(
        n_proc,
        k=p["K"], gSig=list(p["gSig"]),
        merge_thresh=p["merge_thresh"], p=p["p"], dview=dview,
    )
    cnm.params.set("spatial", {"se": np.ones((3, 3, 1), dtype=np.uint8)})

    walls: dict[str, float] = {}

    with PhaseTimer("caiman_offline", "detection_fit", **timer_kwargs) as t:
        cnm.fit(images)
    walls["detection_fit"] = t.wall_s

    cnm.params.change_params(params_dict={
        "fr": p["fr"], "decay_time": p["decay_time"],
        "rval_thr": p["rval_thr"], "min_SNR": p["min_SNR"],
        "use_cnn": p["use_cnn"],
    })

    with PhaseTimer("caiman_offline", "detection_eval", **timer_kwargs) as t:
        cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    walls["detection_eval"] = t.wall_s

    # Refit pass — same step run by the existing fig-comparison-contained
    # pipeline (step2_caiman.py). Re-runs spatial/temporal updates with
    # proper AR(p) deconvolution after components have been filtered.
    cnm.params.set("temporal", {"p": p["p"]})
    with PhaseTimer("caiman_offline", "detection_refit", **timer_kwargs) as t:
        cnm2 = cnm.refit(images)
    walls["detection_refit"] = t.wall_s

    results_dir.mkdir(parents=True, exist_ok=True)
    cnm2.save(str(results_dir / "results.hdf5"))
    return walls


def main() -> None:
    ap = argparse.ArgumentParser(description="CaImAn speed-comparison runner")
    ap.add_argument("--dataset", default=config.DATASET)
    ap.add_argument(
        "--subset", type=int, default=None,
        help="Use only the first N raw TIFs (smoke test).",
    )
    ap.add_argument(
        "--scratch-dir", default="/scratch/caiman",
        help="Working dir for preprocessed TIFs and memmaps.",
    )
    ap.add_argument("--out-csv", default="/results/timings.csv")
    ap.add_argument(
        "--instance", default=os.environ.get("INSTANCE_LABEL", "local"),
    )
    ap.add_argument(
        "--n-cores", type=int, default=16,
        help="Worker processes. Locked to 16 to match the AWS plan.",
    )
    ap.add_argument(
        "--mode", choices=["both", "onacid", "offline"], default="offline",
        help=("Which detection paths to time. Default 'offline' — OnACID "
              "is currently shelved (upstream caiman 1.12.2 doesn't support "
              "3D OnACID; the reviewer's '3D modifications' aren't "
              "reproducible without their attached script)."),
    )
    args = ap.parse_args()

    _cap_blas_threads(1)  # one BLAS thread per worker; cores = parallelism unit

    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    # save_memmap writes relative to CWD by default; CaImAn's internal
    # temp dir defaults to ~/caiman_data. Force both onto the bind-mounted
    # /scratch so we never spill into the container overlay layer.
    os.chdir(scratch)
    os.environ.setdefault("CAIMAN_DATA", str(scratch / "caiman_data"))
    Path(os.environ["CAIMAN_DATA"]).mkdir(parents=True, exist_ok=True)

    prep_dir = scratch / "preprocessed_tifs"
    print(f"CaImAn speed-run on {args.dataset} (mode={args.mode}, "
          f"n_cores={args.n_cores})")
    print(f"  cwd={os.getcwd()}  CAIMAN_DATA={os.environ['CAIMAN_DATA']}")
    tifs = _ensure_preprocessed_tifs(prep_dir, args.subset)
    print(f"  {len(tifs)} preprocessed TIF(s) at {prep_dir}")

    import caiman as cm

    timer_kwargs = dict(
        out_csv=args.out_csv,
        dataset=args.dataset,
        instance=args.instance,
        subset=args.subset,
    )

    c, dview, n_proc = cm.cluster.setup_cluster(
        backend="multiprocessing", n_processes=args.n_cores,
        single_thread=False,
    )

    try:
        # Registration runs ONCE; the same memmap feeds both detection
        # paths. The shared "caiman_motion" row is reused in both
        # tool totals so the table stays coherent.
        memmap_path, mc_wall = _run_registration(
            tifs, dview, timer_kwargs, tool_label="caiman_motion",
        )

        if args.mode in ("both", "onacid"):
            results_dir = Path("/results") / "caiman_onacid" / args.dataset
            det_wall = _run_onacid(
                memmap_path, dview, n_proc, timer_kwargs, results_dir,
            )
            write_total_row(
                args.out_csv, tool="caiman_onacid",
                dataset=args.dataset, instance=args.instance,
                subset=args.subset,
                walls_s={"registration": mc_wall, "detection": det_wall},
            )

        if args.mode in ("both", "offline"):
            results_dir = Path("/results") / "caiman_offline" / args.dataset
            offline_walls = _run_offline(
                memmap_path, dview, n_proc, timer_kwargs, results_dir,
            )
            walls = {"registration": mc_wall, **offline_walls}
            write_total_row(
                args.out_csv, tool="caiman_offline",
                dataset=args.dataset, instance=args.instance,
                subset=args.subset, walls_s=walls,
            )

    finally:
        cm.stop_server(dview=dview)

    print(f"\nDone. Timing rows appended to {args.out_csv}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
