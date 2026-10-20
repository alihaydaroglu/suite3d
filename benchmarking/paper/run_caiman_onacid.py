"""OnACID-3D runner with the p=0 workaround for the 1.12.2 bug.

Background: CaImAn 1.12.2's OnACID.fit_online() for 3D data crashes
inside _prepare_object because cnm.fit() leaves estimates.g / lam / bl
/ neurons_sn as None for 3D (cnmf.py line 569-572, unconditional). The
crash site (online_cnmf.py line 244) is gated by
`if self.params.get('preprocess', 'p'):`, so setting p=0 skips it.
Trade-off: no AR(p) deconvolution / no inferred spikes — F-trace only.

This runner expects an existing C-order memmap path (the registration
output from run_caiman_register.py, or the memmap.txt that run_caiman.py
now writes) so we don't re-do registration. Run with
--memmap-path-file /results/memmap.txt.

Parameters come from `config.CAIMAN_PARAMS` (corrected, round 2);
`--legacy-params` selects the round-1 set, which also re-runs motion
correction inside fit_online (`motion_correct=True`).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import config  # noqa: E402
from run_caiman import _cap_blas_threads, onacid_params_dict  # noqa: E402
from timing_harness import PhaseTimer, write_total_row  # noqa: E402


def _run_onacid_p0(memmap_path, dview, timer_kwargs, results_dir, params, K=None):
    """OnACID 3D with p=0 to dodge the missing-estimates crash.

    Returns dict of phase wall times.
    """
    from caiman.source_extraction.cnmf import params as cnmf_params
    from caiman.source_extraction.cnmf.online_cnmf import OnACID

    opts = cnmf_params.CNMFParams(
        params_dict=onacid_params_dict(params, memmap_path, p=0, K=K))
    cnm = OnACID(params=opts, dview=dview)

    walls: dict[str, float] = {}
    with PhaseTimer(f"caiman_onacid_p0_K{K if K is not None else params['K']}", "detection_online", **timer_kwargs) as t:
        cnm.fit_online()
    walls["detection_online"] = t.wall_s
    n_comp = cnm.estimates.C.shape[0] if cnm.estimates.C is not None else -1
    print(f"OnACID returned {n_comp} components", flush=True)

    results_dir.mkdir(parents=True, exist_ok=True)
    cnm.save(str(results_dir / "results.hdf5"))
    return walls


def main() -> None:
    ap = argparse.ArgumentParser(description="CaImAn OnACID-3D runner (p=0)")
    ap.add_argument("--dataset", default=config.DATASET)
    ap.add_argument("--subset", type=int, default=None,
                    help="Tag the timing rows; informational only here.")
    ap.add_argument("--n-cores", type=int, default=8)
    ap.add_argument("--scratch-dir", default="/scratch/caiman")
    ap.add_argument(
        "--memmap-path-file", default="/results/memmap.txt",
        help="File containing the absolute path to a C-order CaImAn "
             "memmap from a prior registration. The script will not run "
             "MotionCorrect itself.",
    )
    ap.add_argument("--out-csv", default="/results/timings.csv")
    ap.add_argument(
        "--instance", default=os.environ.get("INSTANCE_LABEL", "local"),
    )
    ap.add_argument(
        "--legacy-params", action="store_true",
        help="Run with the parameter set shipped in round 1 "
             "(config.CAIMAN_PARAMS_LEGACY) instead of the corrected set.",
    )
    ap.add_argument(
        "--K", type=int, default=None,
        help="Override the number of components for the OnACID "
             "initialisation (default: config K). 3D OnACID returns exactly "
             "K components, so this sets the size of the timed model. The "
             "published N = 19 bars used K = 200.",
    )
    args = ap.parse_args()

    _cap_blas_threads(1)

    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    os.environ.setdefault("CAIMAN_DATA", str(scratch / "caiman_data"))
    Path(os.environ["CAIMAN_DATA"]).mkdir(parents=True, exist_ok=True)

    params = config.caiman_params(legacy=args.legacy_params)
    memmap_path = Path(args.memmap_path_file).read_text().strip()
    print(f"OnACID-3D (p=0) on {args.dataset} "
          f"(params={'LEGACY' if args.legacy_params else 'corrected'})")
    print(f"  memmap: {memmap_path}")
    K = params["K"] if args.K is None else args.K
    print(f"  n_cores={args.n_cores}, K={K} (whole-FOV init), "
          f"motion_correct={params['motion_correct']}")
    if not Path(memmap_path).exists():
        raise SystemExit(f"Memmap missing: {memmap_path}")

    import caiman as cm

    c, dview, n_proc = cm.cluster.setup_cluster(
        backend="multiprocessing", n_processes=args.n_cores,
        single_thread=False,
    )

    timer_kwargs = dict(
        out_csv=args.out_csv,
        dataset=args.dataset,
        instance=args.instance,
        subset=args.subset,
    )

    try:
        results_dir = Path(args.out_csv).parent / "caiman_onacid_p0" / args.dataset
        walls = _run_onacid_p0(memmap_path, dview, timer_kwargs, results_dir, params, K=args.K)
    finally:
        cm.stop_server(dview=dview)

    walls["registration"] = 0.0  # not done in this script
    write_total_row(
        args.out_csv, tool="caiman_onacid_p0",
        dataset=args.dataset, instance=args.instance,
        subset=args.subset, walls_s=walls,
    )
    print(f"\nDone. OnACID-3D ran in {walls['detection_online']:.1f}s.")


if __name__ == "__main__":
    main()
