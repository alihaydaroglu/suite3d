"""CaImAn speed-comparison runner — both online (OnACID) and offline
(CNMF.fit) detection paths, on a single motion-correction pass.

The reviewer ran approach-1 of `demo_realtime_cnmfE.ipynb` (online
CNMF, CNMF-E swapped for CNMF, 3D, multi-frame parallel). We mirror
that AND also report offline CNMF.fit() on the same data and params,
so the comparison covers both common CaImAn usage modes.

Phases timed (per run):
  registration       — MotionCorrect (NoRMCorre); run once.
  detection_online   — OnACID.fit_online (init batch + online refit).
  detection_offline  — offline CNMF.fit + estimates.evaluate_components
                       + CNMF.refit.

Each "tool" string in the CSV is namespaced so the figures can split:
  caiman_onacid_p0, caiman_offline.

Both share the same registration wall time (same motion-corrected
memmap), which is appended under each tool to keep totals coherent.

Parameters come from `config.CAIMAN_PARAMS` (the corrected round-2
set). `--legacy-params` selects `config.CAIMAN_PARAMS_LEGACY`, the set
shipped with round 1, so the correction can be reproduced.
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


def _n_workers(dview, fallback: int) -> int:
    """Worker count of a multiprocessing.Pool dview (CaImAn's own idiom)."""
    return int(getattr(dview, "_processes", None) or fallback)


def _run_registration(tifs, dview, timer_kwargs, tool_label, params, n_proc=None):
    """Time NoRMCorre under the given tool label. Returns (memmap, wall_s)."""
    import caiman as cm
    from caiman.motion_correction import MotionCorrect
    from caiman.source_extraction.cnmf import params as cnmf_params

    # `fnames` is intentionally omitted from this CNMFParams: when it is
    # present, CNMFParams.change_params() reads T from the files and
    # overwrites splits_rig / splits_els with T // num_frames_split
    # (default 80), i.e. 2 splits for a ~222-frame TIF, so motion
    # correction ran on 2 worker processes whatever --n-cores said. The
    # TIF list is passed straight to MotionCorrect below.
    n_splits = _n_workers(dview, n_proc or 1)
    mc_opts = cnmf_params.CNMFParams(params_dict={
        "strides": params["strides"],
        "overlaps": params["overlaps"],
        "max_shifts": params["max_shifts"],
        "max_deviation_rigid": params["max_deviation_rigid"],
        "pw_rigid": params["pw_rigid"],
        "splits_rig": n_splits,
        "splits_els": n_splits,
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


def onacid_params_dict(params: dict, memmap_path: str, p: int = 0, K: int | None = None) -> dict:
    """Build the CNMFParams dict for OnACID on an already-registered memmap.

    p=0 is the default because CaImAn 1.12.2 cannot run OnACID on 3D
    data otherwise: CNMF.fit leaves estimates.g / lam / bl / neurons_sn
    as None for 3D and OnACID._prepare_object iterates over them; the
    p=0 branch skips that code. With p=0 there is no deconvolution.

    K overrides params["K"] for the OnACID initialisation. In every 3D
    run we have made, OnACID returned exactly K components — the online
    detection never added any — so K sets the size of the model that is
    timed, not an upper bound on detection.
    """
    K = params["K"] if K is None else int(K)
    # The patch parameters (rf / stride / only_init) are deliberately NOT
    # passed: OnACID initialises on the init_batch with K components over the
    # whole FOV and adds components online, which is how the round-1 script
    # ran and how the reviewer measured it. Seeding the init with
    # rf=40 / K=22-per-patch hands the online phase ~60x more components and
    # was measured 8x slower (3053 s vs 372 s on the reviewer's box).
    return {
        "fnames": [memmap_path],
        "fr": params["fr"], "decay_time": params["decay_time"],
        "gSig": params["gSig"], "p": p, "K": K,
        "merge_thr": params["merge_thresh"], "rval_thr": params["rval_thr"],
        "min_SNR": params["min_SNR"], "use_cnn": params["use_cnn"],
        "is3D": True,
        "init_method": params["init_method"],
        "init_batch": params["init_batch"],
        "epochs": params["epochs"], "n_refit": params["n_refit"],
        # Whether fit_online motion-corrects its input again. The memmap
        # it is given has already been through NoRMCorre, so the
        # corrected parameter set turns this off.
        "motion_correct": params["motion_correct"],
        # Motion params are still needed when motion_correct is True.
        "strides": params["strides"], "overlaps": params["overlaps"],
        "max_shifts": params["max_shifts"],
        "max_deviation_rigid": params["max_deviation_rigid"],
        "pw_rigid": params["pw_rigid"],
    }


def _run_onacid(memmap_path, dview, n_proc, timer_kwargs, results_dir, params):
    """OnACID online CNMF (p=0, see onacid_params_dict). Returns wall_s."""
    from caiman.source_extraction.cnmf import params as cnmf_params
    from caiman.source_extraction.cnmf.online_cnmf import OnACID

    opts = cnmf_params.CNMFParams(params_dict=onacid_params_dict(params, memmap_path))
    cnm = OnACID(params=opts, dview=dview)

    with PhaseTimer("caiman_onacid_p0", "detection_online", **timer_kwargs) as t:
        cnm.fit_online()

    results_dir.mkdir(parents=True, exist_ok=True)
    cnm.save(str(results_dir / "results.hdf5"))
    return t.wall_s


def _run_offline(memmap_path, dview, n_proc, timer_kwargs, results_dir, params):
    """Offline CNMF — fit (in patches when rf is set), evaluate, refit.
    Returns dict of sub-phase wall times."""
    import caiman as cm
    from caiman.source_extraction.cnmf.cnmf import CNMF

    p = params
    Yr, dims, T = cm.load_memmap(memmap_path)
    images = np.reshape(Yr.T, [T] + list(dims), order="F")

    cnm = CNMF(
        n_proc,
        k=p["K"], gSig=list(p["gSig"]),
        merge_thresh=p["merge_thresh"], p=p["p"], dview=dview,
        # Patch approach: the FOV is split into patches of side 2*rf
        # (stride = overlap) and n_proc of them are initialised in
        # parallel; K is per patch. rf=None (legacy) runs the whole FOV
        # in one process.
        rf=p["rf"], stride=p["stride"], only_init_patch=p["only_init_patch"],
    )
    cnm.params.set("spatial", {"se": p["se"]})

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
    # dview must be passed explicitly or refit runs single-process.
    cnm.params.set("temporal", {"p": p["p"]})
    with PhaseTimer("caiman_offline", "detection_refit", **timer_kwargs) as t:
        cnm2 = cnm.refit(images, dview=dview)
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
        "--n-cores", type=int, default=8,
        help="Worker processes (default 8, matching the r6a.2xlarge plan).",
    )
    ap.add_argument(
        "--mode", choices=["both", "onacid", "offline"], default="offline",
        help="Which detection paths to time (default: offline).",
    )
    ap.add_argument(
        "--legacy-params", action="store_true",
        help="Run with the parameter set shipped in round 1 "
             "(config.CAIMAN_PARAMS_LEGACY) instead of the corrected set.",
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

    params = config.caiman_params(legacy=args.legacy_params)
    prep_dir = scratch / "preprocessed_tifs"
    print(f"CaImAn speed-run on {args.dataset} (mode={args.mode}, "
          f"n_cores={args.n_cores}, "
          f"params={'LEGACY' if args.legacy_params else 'corrected'})")
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
            params=params, n_proc=n_proc,
        )
        # Persist the memmap path so run_caiman_onacid.py can reuse this
        # registration instead of paying for a second one.
        out_root = Path(args.out_csv).parent
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "memmap.txt").write_text(str(memmap_path) + "\n")

        if args.mode in ("both", "onacid"):
            results_dir = out_root / "caiman_onacid_p0" / args.dataset
            det_wall = _run_onacid(
                memmap_path, dview, n_proc, timer_kwargs, results_dir, params,
            )
            write_total_row(
                args.out_csv, tool="caiman_onacid_p0",
                dataset=args.dataset, instance=args.instance,
                subset=args.subset,
                walls_s={"registration": mc_wall, "detection_online": det_wall},
            )

        if args.mode in ("both", "offline"):
            results_dir = out_root / "caiman_offline" / args.dataset
            offline_walls = _run_offline(
                memmap_path, dview, n_proc, timer_kwargs, results_dir, params,
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
