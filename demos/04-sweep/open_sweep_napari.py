#!/usr/bin/env python
"""Open a Suite3D parameter sweep in the napari sweep viewer.

`curation.SweepUI` lays the sweep out as a grid — one cell per parameter
combination — so you can scrub through planes and see how the correlation map
or the detected ROIs respond to each knob.

Usage
-----
    python open_sweep_napari.py --sweep-dir ./results/s3d-demo-hippocampus/sweeps/corrmap

`--sweep-dir` is the directory containing `sweep_summary.npy`, which
`run_sweep.py` writes.

Requires napari:  pip install "suite3d[napari]"  (or: pip install napari[all])
"""

import argparse
from pathlib import Path

from suite3d.curation import SweepUI


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", required=True,
                   help="directory holding sweep_summary.npy")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    if not (sweep_dir / "sweep_summary.npy").exists():
        raise FileNotFoundError(
            "No sweep_summary.npy in %s — run run_sweep.py first." % sweep_dir
        )

    import napari

    ui = SweepUI(str(sweep_dir))
    ui.load_outputs()
    ui.create_ui()
    napari.run()


if __name__ == "__main__":
    main()
