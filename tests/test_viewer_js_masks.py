"""The JS mask decoder + rasteriser must reproduce the Python footprints.

The viewer paints ROIs by rasterising the RLE into one Int32Array label map per
plane, then picks ROIs by an O(1) pixel lookup. If the JS decoder disagrees with
Python by even one voxel, clicks select the wrong cell. Run the real decode/
rasterise logic under node and diff the label map against numpy.
"""
import json
import os
import shutil
import subprocess
import textwrap

import numpy as np
import pytest

from suite3d.viewer.encode import b64, rle

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

NZ, NY, NX = 4, 40, 48


def _stats(rng, n=25):
    out = []
    for _ in range(n):
        z0 = int(rng.integers(0, NZ))
        y0, x0 = int(rng.integers(0, NY - 9)), int(rng.integers(0, NX - 9))
        zz, yy, xx = [], [], []
        for dz in range(int(rng.integers(1, min(3, NZ - z0) + 1))):
            for dy in range(int(rng.integers(2, 8))):
                for dx in range(int(rng.integers(1, 9))):
                    zz.append(z0 + dz); yy.append(y0 + dy); xx.append(x0 + dx)
        out.append({"coords": np.array([zz, yy, xx]),
                    "lam": rng.random(len(zz)).astype(np.float32) + 1e-3,
                    "peak_val": float(rng.random())})
    return out


def _py_labels(stats):
    """Same rule as the JS: paint ascending peak_val, strongest ROI wins."""
    lab = np.full((NZ, NY, NX), -1, np.int32)
    order = np.argsort([s["peak_val"] for s in stats])
    for roi in order:
        z, y, x = np.asarray(stats[roi]["coords"])
        lab[z, y, x] = roi
    return lab


def test_js_label_map_matches_python():
    rng = np.random.default_rng(11)
    stats = _stats(rng)
    m = rle(stats)
    masks = {k: b64(m[k]) for k in ("off", "z", "y", "x0", "len", "lam")}
    peak = [s["peak_val"] for s in stats]

    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "suite3d", "viewer", "assets", "viewer.js")).read()
    dec_fn = src[src.index("function dec("):src.index("\n}", src.index("function dec(")) + 2]

    body = textwrap.dedent(f"""
        const MASKS = {json.dumps(masks)};
        const NROI = {len(stats)}, NZ = {NZ}, NY = {NY}, NX = {NX};
        const PV = {json.dumps(peak)};
        function atob(s) {{ return Buffer.from(s, 'base64').toString('binary'); }}
        {dec_fn}
        const M = {{off: dec(MASKS.off, Uint32Array), z: dec(MASKS.z, Uint8Array),
                   y: dec(MASKS.y, Uint16Array), x0: dec(MASKS.x0, Uint16Array),
                   len: dec(MASKS.len, Uint8Array), lam: dec(MASKS.lam, Uint8Array)}};
        const labels = [];
        for (let z = 0; z < NZ; z++) labels.push(new Int32Array(NY*NX).fill(-1));
        const order = Array.from({{length: NROI}}, (_, i) => i).sort((a,b) => PV[a]-PV[b]);
        for (const roi of order)
          for (let r = M.off[roi]; r < M.off[roi+1]; r++) {{
            const lab = labels[M.z[r]];
            let idx = M.y[r]*NX + M.x0[r];
            for (let k = 0; k < M.len[r]; k++, idx++) lab[idx] = roi;
          }}
        const flat = [];
        for (let z = 0; z < NZ; z++) flat.push(...labels[z]);
        process.stdout.write(JSON.stringify(flat));
    """)
    out = subprocess.run([NODE, "-e", body], capture_output=True, text=True, check=True)
    got = np.array(json.loads(out.stdout), np.int32).reshape(NZ, NY, NX)
    want = _py_labels(stats)
    assert np.array_equal(got, want), f"{(got != want).sum()} voxels differ"
