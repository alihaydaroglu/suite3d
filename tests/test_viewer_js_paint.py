"""The incremental repaint must never drift from a full repaint.

`diffPaint` repaints only the ROIs whose appearance changed -- that is what keeps
a filter drag at ~0.2 ms instead of ~3 ms on SS004's 682k-pixel planes. The failure
mode is silent and cumulative: an ROI stays lit after it was filtered out, or a
deselected ROI keeps its white highlight, and nothing throws. So drive the real
`fullPaint`/`diffPaint` from assets/viewer.js through a long sequence of filter,
selection and hover changes, and assert the pixel buffer still equals a full
repaint computed from scratch.

Skipped when node is unavailable.
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

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "suite3d", "viewer", "assets")
NZ, NY, NX = 3, 36, 44


def _stats(rng, n=30):
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


def _fn(src, name):
    start = src.index(f"function {name}(")
    return src[start:src.index("\n}", start) + 2]


def test_diff_paint_matches_full_paint():
    rng = np.random.default_rng(23)
    stats = _stats(rng)
    n = len(stats)
    m = rle(stats)
    masks = {k: b64(m[k]) for k in ("off", "z", "y", "x0", "len", "lam")}
    peak = [s["peak_val"] for s in stats]

    src = open(os.path.join(ASSETS, "viewer.js")).read()
    fns = "\n".join(_fn(src, f) for f in ("dec", "paintRun", "paintROI", "fullPaint",
                                          "diffPaint", "codeOf"))

    # A scripted session: filter ROIs out and back in, select, hover, and combine
    # them -- every transition between the four appearance codes.
    rng2 = np.random.default_rng(5)
    steps = []
    for i in range(40):
        # Plane changes and "only selected" toggles force a full repaint, which
        # reseeds the cache -- so keep them RARE. Long runs of incremental-only
        # steps are what let a stale cache actually drift.
        steps.append({"visible": rng2.integers(0, 2, n).tolist(),
                      "selected": int(rng2.integers(0, n)) if i % 5 else -1,
                      "hover": int(rng2.integers(-1, n)),
                      "plane": (i // 13) % NZ,
                      "onlySelected": bool((i // 17) % 2)})

    body = textwrap.dedent(f"""
        const MASKS = {json.dumps(masks)};
        const NROI = {n}, NZ = {NZ}, NY = {NY}, NX = {NX};
        const PV = {json.dumps(peak)};
        const STEPS = {json.dumps(steps)};
        function atob(s) {{ return Buffer.from(s, 'base64').toString('binary'); }}
        const HIDDEN = 0, NORMAL = 1, SELECTED = 2, HOVER = 3;
        {_fn(src, "dec")}
        const M = {{off: dec(MASKS.off, Uint32Array), z: dec(MASKS.z, Uint8Array),
                   y: dec(MASKS.y, Uint16Array), x0: dec(MASKS.x0, Uint16Array),
                   len: dec(MASKS.len, Uint8Array), lam: dec(MASKS.lam, Uint8Array)}};
        const NRUN = M.z.length;
        const LUT = new Uint8Array(NROI*3);
        for (let i = 0; i < NROI*3; i++) LUT[i] = (i * 37) % 200 + 20;

        const labels = [], alpha = [], planeRuns = [];
        const roiOf = new Int32Array(NRUN);
        for (let z = 0; z < NZ; z++) {{ labels.push(new Int32Array(NY*NX).fill(-1)); alpha.push(new Uint8Array(NY*NX)); }}
        for (let roi = 0; roi < NROI; roi++) for (let r = M.off[roi]; r < M.off[roi+1]; r++) roiOf[r] = roi;
        const order = Array.from({{length: NROI}}, (_, i) => i).sort((a,b) => PV[a]-PV[b]);
        for (const roi of order) for (let r = M.off[roi]; r < M.off[roi+1]; r++) {{
          const lab = labels[M.z[r]], al = alpha[M.z[r]];
          let idx = M.y[r]*NX + M.x0[r];
          for (let k = 0; k < M.len[r]; k++, idx++) {{ lab[idx] = roi; al[idx] = M.lam[r]; }}
        }}
        const cnt = new Int32Array(NZ); for (let r=0;r<NRUN;r++) cnt[M.z[r]]++;
        for (let z=0;z<NZ;z++) planeRuns.push(new Int32Array(cnt[z]));
        const fillp = new Int32Array(NZ); for (let r=0;r<NRUN;r++) {{ const z=M.z[r]; planeRuns[z][fillp[z]++]=r; }}

        let visible = new Uint8Array(NROI);
        const drawn = new Uint8Array(NROI);
        const state = {{plane: 1, selected: -1, hover: -1, onlySelected: false}};
        const img = {{data: new Uint8ClampedArray(NY*NX*4)}};
        {fns}

        fullPaint();
        let bad = 0;
        for (const s of STEPS) {{
          visible = Uint8Array.from(s.visible);
          state.selected = s.selected; state.hover = s.hover;
          const wasOnly = state.onlySelected, wasPlane = state.plane;
          state.onlySelected = s.onlySelected; state.plane = s.plane;
          // the app does a full repaint when the plane changes or "only selected"
          // toggles; everything else must survive on the incremental path alone
          if (wasOnly !== s.onlySelected || wasPlane !== s.plane) fullPaint();
          else diffPaint();

          const inc = Uint8ClampedArray.from(img.data);
          const drawnSnap = Uint8Array.from(drawn);

          fullPaint();                          // ground truth, from scratch
          const truth = Uint8ClampedArray.from(img.data);

          // Put the incremental state back, or fullPaint's rewrite of `drawn`
          // would repair the very drift we are hunting, and the test would pass
          // even with the cache update deleted.
          img.data.set(inc); drawn.set(drawnSnap);

          // What is *seen*: alpha everywhere, and colour only where alpha > 0.
          // A cleared pixel keeps stale RGB under alpha=0; that is invisible, and
          // demanding byte equality there would only be testing fill(0).
          for (let p = 0; p < inc.length; p += 4) {{
            if (inc[p+3] !== truth[p+3]) {{ bad++; break; }}
            if (truth[p+3] === 0) continue;
            if (inc[p] !== truth[p] || inc[p+1] !== truth[p+1] || inc[p+2] !== truth[p+2]) {{ bad++; break; }}
          }}
        }}
        console.log(bad);
    """)
    out = subprocess.run([NODE, "-e", body], capture_output=True, text=True, check=True)
    assert int(out.stdout.strip()) == 0, f"{out.stdout.strip()}/40 steps drifted from a full repaint"
