#!/usr/bin/env python
"""Suite3D demo 02 — Light-beads microscopy, ~40k neurons (SS004).

The large one. 22 planes over a 5-minute recording, and the reason Suite3D
exists: a volume this dense is where 3D detection pulls decisively away from
plane-by-plane 2D segmentation.

LBM data needs three things no conventional 2P recording does, all handled in
`common/datasets.py`:

* **Strip fusion.** The mesoscope writes several ROI strips per plane, which are
  fused back into one image (`fuse_strips`, `fuse_shift_override`).
* **Crosstalk subtraction.** The two cavities are interleaved in depth and bleed
  into one another (`subtract_crosstalk`, `cavity_size=13`).
* **Plane reordering.** ScanImage stores the cavities interleaved; `planes`
  maps them back into depth order. This mapping is specific to the microscope.

It also has lower per-voxel SNR than a conventional recording, so it runs a
gentler intensity gate (`intensity_thresh=1.0`), a coarser detection time bin,
and a lower voxel-SNR floor for ROI inclusion.

Data
----
`<data-root>/lbm/raw/*.tif` — 13 tifs, 56.2 GB. The registered movie is ~38 GB,
so make sure --out-dir has room. This demo is slow; that is expected.

Usage
-----
    python run_pipeline.py --data-root /path/to/figshare --out-dir ./results

Expect roughly 43,000 ROIs. Every **detection and segmentation** parameter here
is pinned to the values that produced the published 43,652-ROI segmentation of
this recording. Registration is not: this demo uses nonrigid registration, where
the published run was rigid-only. Registration feeds the correlation map, so the
count will land near 43,652 rather than exactly on it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import main   # noqa: E402


if __name__ == "__main__":
    main("lbm", __doc__)
