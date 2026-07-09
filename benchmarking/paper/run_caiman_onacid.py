"""OnACID-3D runner with the p=0 workaround for the 1.12.2 bug.

Background: CaImAn 1.12.2's OnACID.fit_online() for 3D data crashes
inside _prepare_object because cnm.fit() leaves estimates.g / lam / bl
/ neurons_sn as None for 3D (cnmf.py line 569-572, unconditional). The
crash site (online_cnmf.py line 244) is gated by
`if self.params.get('preprocess', 'p'):`, so setting p=0 skips it.
Trade-off: no AR(p) deconvolution / no inferred spikes — F-trace only.

This runner expects an existing C-order memmap path (the registration
output from run_caiman_register.py or a prior full run) so we don't
re-do registration. Run with --memmap-path-file /results/memmap.txt.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import config  # noqa: E402
from run_caiman import _cap_blas_threads  # noqa: E402
from timing_harness import PhaseTimer, write_total_row  # noqa: E402


def _run_onacid_p0(memmap_path, dview, timer_kwargs, results_dir):
    """OnACID 3D with p=0 to dodge the missing-estimates crash.

    Returns dict of phase wall times.
    """
    from caiman.source_extraction.cnmf import params as cnmf_params
    from caiman.source_extraction.cnmf.online_cnmf import OnACID

    p = config.CAIMAN_PARAMS
    opacid_dict = {
        "fnames": [memmap_path],
        "fr": p["fr"], "decay_time": p["decay_time"],
        "gSig": p["gSig"],
        "p": 0,                            # ← workaround
        "K": p["K"],
        "merge_thr": p["merge_thresh"], "rval_thr": p["rval_thr"],
        "min_SNR": p["min_SNR"], "use_cnn": p["use_cnn"],
        "is3D": True,
        "init_method": p["init_method"],    # "cnmf"
        "init_batch": p["init_batch"],      # 200 frames offline init
        "epochs": p["epochs"], "n_refit": p["n_refit"],
    }
    opts = cnmf_params.CNMFParams(params_dict=opacid_dict)
    cnm = OnACID(params=opts, dview=dview)

    walls: dict[str, float] = {}
    with PhaseTimer("caiman_onacid_p0", "detection_online", **timer_kwargs) as t:
        cnm.fit_online()
    walls["detection_online"] = t.wall_s

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
    args = ap.parse_args()

    _cap_blas_threads(1)

    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    os.environ.setdefault("CAIMAN_DATA", str(scratch / "caiman_data"))
    Path(os.environ["CAIMAN_DATA"]).mkdir(parents=True, exist_ok=True)

    memmap_path = Path(args.memmap_path_file).read_text().strip()
    print(f"OnACID-3D (p=0) on {args.dataset}")
    print(f"  memmap: {memmap_path}")
    print(f"  n_cores={args.n_cores}, K={config.CAIMAN_PARAMS['K']}")
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
        results_dir = Path("/results") / "caiman_onacid_p0" / args.dataset
        walls = _run_onacid_p0(memmap_path, dview, timer_kwargs, results_dir)
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
