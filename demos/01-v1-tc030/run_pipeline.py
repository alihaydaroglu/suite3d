#!/usr/bin/env python
"""Suite3D demo 01 — V1, standard multi-plane 2P (TC030).

The simplest of the three demos, and the one to read first. A 7-plane
conventional two-photon recording of mouse visual cortex.

What makes it a good starting point: it needs **no parameter overrides at all**
beyond the acquisition geometry. Every filtering, detection and extraction
parameter is a Suite3D default. If you want to know what Suite3D does out of
the box, this is it.

Data
----
Download the figshare archive and point --data-root at it. This demo reads
`<data-root>/v1/raw/*.tif` (10 tifs, 21.1 GB).

Usage
-----
    python run_pipeline.py --data-root /path/to/figshare --out-dir ./results

    # re-open the viewer on a finished run, skipping the compute
    python run_pipeline.py --out-dir ./results --skip-init --skip-register \
        --skip-corrmap --skip-segment --skip-extract --viewer napari

Expect ~845 ROIs. Registration dominates the wall time; see demos/README.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import main   # noqa: E402


if __name__ == "__main__":
    main("v1", __doc__)
