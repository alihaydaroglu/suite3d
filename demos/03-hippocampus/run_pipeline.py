#!/usr/bin/env python
"""Suite3D demo 03 — densely packed CA1, standard 2P (ATL020).

A 4-plane hippocampal recording. Densely packed somata make this the hardest of
the three for segmentation, and the shallow stack makes it the most interesting
for registration.

The one override worth understanding: **`apply_z_shift=False`**.

Suite3D measures rigid motion in 3D, including an axial (z) component. On a
4-plane stack the axial phase-correlation search window is only +-2 planes, and
6.6% of this recording's volumes peg at that cap — the estimate saturates rather
than converging. Applying a saturated estimate would translate those frames by
62% of the volume depth. So z is still *measured* in 3D (`3d_reg=True`), it is
simply not *applied*.

That is the documented remedy for shallow stacks. Note it is not the same as
turning off 3D registration: `3d_reg` stays on here, and should stay on.

Data
----
`<data-root>/hippocampus/raw/*.tif` — 10 tifs, 21.1 GB.

Usage
-----
    python run_pipeline.py --data-root /path/to/figshare --out-dir ./results

Expect ~1,347 ROIs. See `../04-sweep/` for how the detection parameters were
chosen on this recording.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import main   # noqa: E402


if __name__ == "__main__":
    main("hippocampus", __doc__)
