"""The JS and Python `iscell` predicates must agree exactly.

`Job.import_curation()` recomputes iscell in Python from the same curation.json
the browser produced. If the two predicates drift, the user's downloaded
iscell.npy silently disagrees with what they curated on screen. So run the real
assets/viewer.js `applyCuration` under node and diff it against
suite3d.viewer.curation.apply_curation.

Skipped when node is unavailable.
"""
import base64
import json
import os
import shutil
import subprocess
import textwrap

import numpy as np
import pytest

from suite3d.viewer import curation as cur
from suite3d.viewer.encode import b64

NODE = shutil.which("node")
ASSETS = os.path.join(os.path.dirname(cur.__file__), "assets")

pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


# The curation block of viewer.js, extracted so we test the shipped source rather
# than a copy. If the markers move, this test fails loudly instead of silently
# testing nothing.
def _extract_apply_curation():
    src = open(os.path.join(ASSETS, "viewer.js")).read()
    start = src.index("function applyCuration()")
    end = src.index("\n}", start) + 2
    return src[start:end]


def _run_js(stats, filters, manual, n_rois):
    body = textwrap.dedent(f"""
        const NROI = {n_rois};
        const STAT = {json.dumps({k: v.tolist() for k, v in stats.items()})};
        const state = {{filters: {json.dumps(filters)},
                        manual: new Map({json.dumps([[int(k), int(v)] for k, v in manual.items()])})}};
        let visible = new Uint8Array(NROI);
        {_extract_apply_curation()}
        applyCuration();
        console.log(JSON.stringify(Array.from(visible)));
    """)
    out = subprocess.run([NODE, "-e", body], capture_output=True, text=True, check=True)
    return np.array(json.loads(out.stdout), dtype=bool)


def test_apply_curation_matches_js():
    rng = np.random.default_rng(7)
    n = 400
    stats = {
        "npix": rng.integers(4, 2000, n).astype(np.float32),
        "zspan": rng.integers(1, 20, n).astype(np.float32),
        "peak_val": rng.random(n).astype(np.float32),
        "vox_snr": rng.random(n).astype(np.float32),
    }
    # NaNs must never fail a filter, on either side
    stats["peak_val"][::37] = np.nan
    stats["vox_snr"][5::53] = np.nan

    filters = {"npix": [50, 1500], "zspan": [2, 12],
               "peak_val": [0.2, 0.9], "vox_snr": [0.1, 0.95]}
    manual = {3: 1, 11: 0, 250: 1, 399: 0}

    want = cur.apply_curation(stats, filters, manual, n_rois=n)
    got = _run_js(stats, filters, manual, n)
    assert np.array_equal(want, got), f"{(want != got).sum()} ROIs disagree"


def test_iscell_npy_written_by_js_loads_in_numpy():
    """The browser builds iscell.npy bytes by hand; numpy must accept them."""
    src = open(os.path.join(ASSETS, "viewer.js")).read()
    start = src.index("function iscellNpy()")
    end = src.index("\n}", start) + 2
    n = 137
    rng = np.random.default_rng(3)
    vis = rng.integers(0, 2, n).astype(np.uint8)
    body = textwrap.dedent(f"""
        const NROI = {n};
        const visible = Uint8Array.from({json.dumps(vis.tolist())});
        {src[start:end]}
        process.stdout.write(Buffer.from(iscellNpy()).toString('base64'));
    """)
    out = subprocess.run([NODE, "-e", body], capture_output=True, text=True, check=True)
    import io
    arr = np.load(io.BytesIO(base64.b64decode(out.stdout)))
    assert arr.dtype == np.bool_
    assert np.array_equal(arr, vis.astype(bool))
