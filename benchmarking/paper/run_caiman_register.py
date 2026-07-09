"""Registration-only CaImAn runner.

Runs NoRMCorre motion correction and persists the resulting C-order
memmap path to /results/memmap.txt so a separate detection-only run
can be replayed against it.

Imports the existing _run_registration helper from run_caiman.py so
the registration logic stays single-sourced.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import config  # noqa: E402
from run_caiman import (  # noqa: E402
    _cap_blas_threads,
    _ensure_preprocessed_tifs,
    _run_registration,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="CaImAn registration-only runner")
    ap.add_argument("--dataset", default=config.DATASET)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--n-cores", type=int, default=8)
    ap.add_argument("--scratch-dir", default="/scratch/caiman")
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

    prep_dir = scratch / "preprocessed_tifs"
    tifs = _ensure_preprocessed_tifs(prep_dir, args.subset)
    print(f"{len(tifs)} preprocessed TIF(s) at {prep_dir}")

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
        memmap_path, mc_wall = _run_registration(
            tifs, dview, timer_kwargs, tool_label="caiman_motion",
        )
    finally:
        cm.stop_server(dview=dview)

    memmap_path_file = Path("/results") / "memmap.txt"
    memmap_path_file.write_text(str(memmap_path) + "\n")
    print(f"Memmap path saved to {memmap_path_file}: {memmap_path}")
    print(f"Registration wall: {mc_wall:.1f}s")


if __name__ == "__main__":
    main()
