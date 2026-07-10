"""Binary encoders for the portable HTML viewer.

Everything here produces base64 strings that JS decodes into typed arrays.  The
viewer runs from `file://`, where fetch()/XHR are blocked, so all payloads travel
inside `.js` files loaded via <script src>.  Keeping them base64 (rather than JSON
numbers) is what makes 43k ROIs affordable: a JSON array of 550k run records is
~15 MB of text, the same data as packed bytes + base64 is ~5 MB.

No dependencies beyond numpy + stdlib.
"""

import base64
import io as _io

import numpy as np


def b64(arr):
    """Base64 of an array's raw little-endian bytes."""
    a = np.ascontiguousarray(arr)
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    return base64.b64encode(a.tobytes()).decode("ascii")


def npy_bytes(arr):
    """Serialise an array to .npy bytes.

    Used so the browser can hand back a real `iscell.npy` via a download link --
    the page cannot write files in place, but it can build valid .npy bytes.
    Python side keeps this here so the JS writer can be diffed against it in tests.
    """
    buf = _io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return buf.getvalue()


def rle(stats, max_run=255):
    """Run-length encode every ROI footprint along x.

    Returns a dict of structure-of-arrays.  Runs are emitted in ROI order, so a
    per-run `roi` column would be redundant -- `off` (n_rois + 1) reconstructs it
    in JS and saves 2 bytes per run.

        off  uint32 (n_rois+1)   run index where each ROI starts
        z    uint8              plane
        y    uint16             row
        x0   uint16             first column of the run
        len  uint8              run length in px (runs > max_run are split)
        lam  uint8              mean footprint weight over the run, 1..255

    ~7 bytes/run.  SS004 (43,652 ROIs) -> 550k runs -> 3.9 MB raw, 5.2 MB base64.

    Vectorised: the prototype in pub/fig-datasets-rerun/make_viewer.py looped in
    Python over every run, which costs minutes on the LBM job.
    """
    offs = np.zeros(len(stats) + 1, dtype=np.uint32)
    zz, yy, xx, ll, mm = [], [], [], [], []
    total = 0

    for i, s in enumerate(stats):
        c = np.asarray(s["coords"])
        if c.shape[0] != 3:
            c = c.T
        z, y, x = (c[k].astype(np.int64) for k in range(3))
        lam = np.asarray(s["lam"], dtype=np.float64)

        order = np.lexsort((x, y, z))
        z, y, x, lam = z[order], y[order], x[order], lam[order]

        peak = lam.max()
        lam8 = np.clip(np.round(255.0 * lam / (peak if peak > 0 else 1.0)), 1, 255)

        # a run breaks wherever z or y changes, or x is not contiguous
        brk = np.ones(len(x), dtype=bool)
        if len(x) > 1:
            brk[1:] = (z[1:] != z[:-1]) | (y[1:] != y[:-1]) | (x[1:] != x[:-1] + 1)
        starts = np.flatnonzero(brk)
        ends = np.append(starts[1:], len(x))
        lens = ends - starts

        # split runs longer than max_run (len is uint8).  Build the split offsets
        # without a Python loop over runs.
        nsplit = -(-lens // max_run)                      # ceil division
        run_id = np.repeat(np.arange(len(starts)), nsplit)
        within = np.arange(nsplit.sum()) - np.repeat(np.cumsum(nsplit) - nsplit, nsplit)
        seg_start = starts[run_id] + within * max_run
        seg_len = np.minimum(lens[run_id] - within * max_run, max_run)

        # mean lam over each emitted segment, via a prefix sum
        cs = np.concatenate([[0.0], np.cumsum(lam8)])
        seg_lam = (cs[seg_start + seg_len] - cs[seg_start]) / seg_len

        zz.append(z[seg_start]); yy.append(y[seg_start]); xx.append(x[seg_start])
        ll.append(seg_len); mm.append(seg_lam)
        total += len(seg_start)
        offs[i + 1] = total

    cat = lambda parts, dt: (np.concatenate(parts).astype(dt) if parts
                             else np.zeros(0, dt))
    return dict(
        off=offs,
        z=cat(zz, np.uint8),
        y=cat(yy, np.uint16),
        x0=cat(xx, np.uint16),
        len=cat(ll, np.uint8),
        lam=np.clip(np.round(cat(mm, np.float64)), 1, 255).astype(np.uint8),
        n_runs=total,
    )


def quantize_traces(F, dtype="int16"):
    """Per-ROI affine quantisation:  value = q * scale + offset.

    Returns (q, scale, offset).  Visually lossless for display and half the size of
    float32.  A flat trace gets scale=1 so dequantisation is exact.
    `dtype="float32"` passes the data through unchanged (scale=1, offset=0).
    """
    F = np.asarray(F, dtype=np.float64)
    if dtype == "float32":
        n = F.shape[0]
        return F.astype(np.float32), np.ones(n, np.float32), np.zeros(n, np.float32)
    if dtype != "int16":
        raise ValueError(f"trace_dtype must be 'int16' or 'float32', got {dtype!r}")

    lo = F.min(axis=1)
    hi = F.max(axis=1)
    span = hi - lo
    # Flat rows are common and must not be mangled: `spks` is all-zero for any ROI
    # with no detected events. They encode as q=0, scale=1, offset=value.
    flat = span <= 0
    span[flat] = 1.0                      # avoid 0/0
    scale = span / 65534.0                # int16 range, one code held in reserve
    q = np.round((F - lo[:, None]) / scale[:, None]) - 32767
    q = np.clip(q, -32767, 32767).astype(np.int16)
    offset = lo + 32767.0 * scale

    q[flat] = 0
    scale[flat] = 1.0
    offset[flat] = lo[flat]
    return q, scale.astype(np.float32), offset.astype(np.float32)
