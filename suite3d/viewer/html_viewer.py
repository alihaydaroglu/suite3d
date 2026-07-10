"""Write a portable, offline HTML viewer for one Suite3D job's ROIs.

The output directory can be copied to any machine and opened by double-clicking
`viewer.html`.  No Python, no server, no internet.  See
dev/coordination/specs/html-viewer.md for the full design.

Nothing here adds a dependency: numpy, matplotlib (already required) and stdlib.
"""

import json
import os
import shutil
import warnings

import numpy as np

from . import curation as cur
from .encode import b64, quantize_traces, rle
from .movies import normalize_spec, write_movie_snippets

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

HTML = """<!doctype html>
<meta charset="utf-8">
<title>Suite3D — __JOB__</title>
<link rel="stylesheet" href="viewer/viewer.css">
<div id="app">
  <div id="side">
    <h1 id="title"></h1>
    <div class="sub" id="sub"></div>
    <div class="warn" id="fswarn"></div>

    <h2>View</h2>
    <div class="row"><label>plane</label><input type="range" id="plane" min="0" step="1"><output id="hudp"></output></div>
    <div class="row"><label>background</label>
      <select id="bgsel"><option value="corrmap">correlation map</option><option value="mean">mean image</option></select>
    </div>
    <div class="row"><label>only selected</label><input type="checkbox" id="only"></div>
    <div class="btnrow"><button id="fit">Fit</button></div>

    <div id="moviebox" hidden>
      <h2>Movie</h2>
      <div class="row"><label>source</label><select id="movsel"></select></div>
      <div class="row"><label>frame</label><input type="range" id="frame" min="0" step="1"><output id="frameout"></output></div>
      <div class="row"><label>fps</label><input type="range" id="fps" min="1" max="60" step="1" value="12"><output id="fpsout">12</output></div>
      <div class="btnrow"><button id="play">Play</button><button id="movoff">Hide</button></div>
      <div class="hint" id="movhint"></div>
    </div>

    <h2>Filters</h2>
    <div id="filters"></div>
    <div id="counts"></div>
    <div class="hint">Drag either end of a band; double-click a plot to reset it.
      Bars show how the ROIs are distributed.</div>
    <div class="hint">Click an ROI to see its traces. <kbd>Alt</kbd>+click toggles it
      cell / not-cell. <kbd>↑</kbd><kbd>↓</kbd> change plane, <kbd>f</kbd> fits,
      <kbd>space</kbd> plays, <kbd>Esc</kbd> deselects.</div>

    <h2>Curation</h2>
    <div class="btnrow">
      <button id="dlnpy" class="primary">Download iscell.npy</button>
      <button id="dljson">curation.json</button>
      <button id="reset">Reset</button>
    </div>
    <div class="hint">Filters + manual overrides are saved in this browser and
      restored next time. <code>job.import_curation()</code> applies the JSON back
      to the run.</div>
  </div>

  <div id="stage">
    <div id="layers"><img id="bg" alt=""><img id="mov" alt="" hidden><canvas id="ov"></canvas></div>
    <div id="hud"></div>
  </div>

  <div id="tracebar">
    <div id="roihdr">
      <div class="sw" id="sw"></div>
      <span class="k">ROI</span><span class="v" id="roiid"></span>
      <span class="k">voxels</span><span class="v" id="npix"></span>
      <span class="k">z-planes</span><span class="v" id="zspan"></span>
      <span class="k">peak</span><span class="v" id="pv"></span>
      <span class="k">status</span><span class="v" id="incl"></span>
      <span class="grow"></span>
    </div>
    <div id="tracemsg"></div>
    <canvas id="traces"></canvas>
  </div>
</div>
<script src="viewer/meta.js"></script>
<script src="viewer/masks.js"></script>
<script src="viewer/viewer.js"></script>
"""


def _resolve_fs_vol(params, n_planes, fs_vol=None):
    """Volume rate, with a loud complaint when params['fs'] is the plane rate.

    `get_vol_rate` reads the per-plane rate for some acquisitions, so
    `params['fs']` cannot be trusted blindly: s3d-TC040_2025-10-28 stores
    fs=30.019 on a 7-plane recording (true volume rate 4.288 Hz).  Guessing would
    silently mislabel every time axis by n_planes, so warn rather than correct.
    """
    if fs_vol is not None:
        return float(fs_vol), ""
    fs = float(params.get("fs", 1.0))
    msg = ""
    if n_planes > 1 and fs > 20.0:
        implied = fs / n_planes
        msg = (f"params['fs']={fs:.3f} Hz looks like the PLANE rate for a "
               f"{n_planes}-plane volume (implied volume rate {implied:.3f} Hz). "
               f"Time axes may be wrong by {n_planes}x. Pass fs_vol=... to fix.")
        warnings.warn(msg, stacklevel=3)
    return fs, msg


def _plane_pngs(out_dir, vmap, mean_img, nz):
    """Pre-colormapped PNGs, one per plane per background.

    Colormapping happens here, in Python, because the page must never call
    getImageData on these: a file:// <img> taints the canvas.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for name, vol in (("corrmap", vmap), ("mean", mean_img)):
        if vol is None:
            continue
        for z in range(nz):
            img = np.asarray(vol[z], dtype=np.float64)
            fin = img[np.isfinite(img)]
            if fin.size == 0:
                fin = np.zeros(1)
            lo, hi = np.percentile(fin, [1, 99.5])
            if hi <= lo:
                hi = lo + 1e-6
            mpimg.imsave(os.path.join(out_dir, f"z{z:02d}_{name}.png"),
                         np.clip((img - lo) / (hi - lo), 0, 1),
                         cmap="magma" if name == "corrmap" else "gray", vmin=0, vmax=1)
            written += 1
    return written


def _write_traces(out_dir, traces, chunk_rois, trace_dtype):
    """Shard traces into JSONP chunks: traces/chunk_0007.js -> S3D.trace(7, {...}).

    <script src> is not CORS-restricted on file://, so injecting one of these on
    click is how the viewer loads a trace lazily without a server.  Inlining is
    not an option: SS004's F+Fneu+spks are 681 MB as float32.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_rois = next(iter(traces.values())).shape[0]
    n_chunks = -(-n_rois // chunk_rois)
    total = 0
    for c in range(n_chunks):
        s = slice(c * chunk_rois, min((c + 1) * chunk_rois, n_rois))
        payload = {"n": int(s.stop - s.start), "nt": int(next(iter(traces.values())).shape[1])}
        for key, arr in traces.items():
            q, sc, off = quantize_traces(arr[s], trace_dtype)
            payload[key] = b64(q)
            payload[key + "_scale"] = b64(sc)
            payload[key + "_offset"] = b64(off)
        p = os.path.join(out_dir, f"chunk_{c:04d}.js")
        with open(p, "w") as f:
            f.write(f"S3D.trace({c},{json.dumps(payload)});\n")
        total += os.path.getsize(p)
    return n_chunks, total


def make_html_viewer(job, export_path=None, result_dir_name="rois", traces="auto",
                     trace_dtype="int16", chunk_rois=256, fs_vol=None,
                     movie_snippet=None):
    """Build the portable viewer.  See Job.make_html_viewer for the docstring."""
    spec = normalize_spec(movie_snippet) if movie_snippet else None

    jd = job.dirs["job_dir"]
    roi_dir = os.path.join(jd, result_dir_name)
    stats_p = os.path.join(roi_dir, "stats.npy")
    if not os.path.exists(stats_p):
        raise FileNotFoundError(f"no segmentation at {stats_p}")

    export_path = export_path or os.path.join(jd, f"s3d-results-{job.job_id}")
    vdir = os.path.join(export_path, "viewer")
    os.makedirs(vdir, exist_ok=True)

    stats = np.load(stats_p, allow_pickle=True)
    n_rois = len(stats)
    job.log(f"make_html_viewer: {n_rois} ROIs -> {export_path}", 1)

    vmap = None
    vp = os.path.join(jd, "corrmap", "vmap.npy")
    if os.path.exists(vp):
        vmap = np.load(vp)
    nz = vmap.shape[0] if vmap is not None else int(max(np.asarray(s["coords"])[0].max() for s in stats) + 1)

    mean_img = None
    sp = os.path.join(jd, "summary", "summary.npy")
    if os.path.exists(sp):
        summ = np.load(sp, allow_pickle=True).item()
        for k in ("ref_img_3d", "mean_img"):
            if k in summ and summ[k] is not None:
                mean_img = np.asarray(summ[k])
                break

    shape = (nz,) + (vmap.shape[1:] if vmap is not None else
                     (int(max(np.asarray(s["coords"])[1].max() for s in stats)) + 1,
                      int(max(np.asarray(s["coords"])[2].max() for s in stats)) + 1))

    # ---- masks
    m = rle(stats)
    with open(os.path.join(vdir, "masks.js"), "w") as f:
        f.write("window.S3D_MASKS=" + json.dumps(
            {k: b64(m[k]) for k in ("off", "z", "y", "x0", "len", "lam")}) + ";\n")
    job.log(f"  masks: {m['n_runs']} runs", 2)

    # ---- traces
    have_F = os.path.exists(os.path.join(roi_dir, "F.npy"))
    want = have_F if traces == "auto" else bool(traces)
    n_chunks, nt = 0, 0
    if want:
        if not have_F:
            raise FileNotFoundError(f"traces requested but {roi_dir}/F.npy is missing")
        tr = {}
        for k in ("F", "Fneu", "spks"):
            p = os.path.join(roi_dir, f"{k}.npy")
            if os.path.exists(p):
                tr[k] = np.load(p)
        nt = int(next(iter(tr.values())).shape[1])
        n_chunks, nbytes = _write_traces(os.path.join(vdir, "traces"), tr, chunk_rois, trace_dtype)
        job.log(f"  traces: {n_chunks} chunks, {nbytes/1e6:.1f} MB ({trace_dtype})", 2)

    # ---- planes
    npng = _plane_pngs(os.path.join(vdir, "planes"), vmap, mean_img, nz)
    job.log(f"  planes: {npng} png", 2)

    # ---- movie snippets (never the full movie: the export dir must stay copyable)
    movies = write_movie_snippets(job, vdir, spec, nz, shape[1], shape[2],
                                  nt_trace=nt) if spec else {}

    # ---- meta
    fs, fs_msg = _resolve_fs_vol(job.params, nz, fs_vol)
    table = cur.stat_table(stats)
    meta = dict(
        job_id=str(job.job_id), n_rois=int(n_rois), shape=[int(x) for x in shape],
        voxel_size_um=[float(x) for x in job.params.get("voxel_size_um", (1, 1, 1))],
        fs_vol=float(fs), fs_warning=fs_msg, n_frames=int(nt),
        has_traces=bool(want), n_chunks=int(n_chunks), chunk_rois=int(chunk_rois),
        trace_dtype=trace_dtype, movies=movies,
        stats={k: b64(v) for k, v in table.items()},
    )
    with open(os.path.join(vdir, "meta.js"), "w") as f:
        f.write("window.S3D_META=" + json.dumps(meta) + ";\n")

    # ---- static assets + page
    for a in ("viewer.css", "viewer.js"):
        shutil.copyfile(os.path.join(ASSETS, a), os.path.join(vdir, a))
    with open(os.path.join(export_path, "viewer.html"), "w") as f:
        f.write(HTML.replace("__JOB__", str(job.job_id)))

    job.log(f"make_html_viewer: open {os.path.join(export_path, 'viewer.html')}", 1)
    return os.path.join(export_path, "viewer.html")


def import_curation(job, path, result_dir_name="rois", write=True):
    """Apply a `curation.json` downloaded from the viewer back into the job dir."""
    d = cur.load_curation(path)
    stats = np.load(os.path.join(job.dirs["job_dir"], result_dir_name, "stats.npy"),
                    allow_pickle=True)
    if len(stats) != d["n_rois"]:
        raise ValueError(f"curation has {d['n_rois']} ROIs, run has {len(stats)}")
    table = cur.stat_table(stats)
    iscell = cur.apply_curation(table, d.get("filters"),
                                {int(k): v for k, v in d.get("manual", {}).items()},
                                n_rois=len(stats))
    if write:
        p = os.path.join(job.dirs["job_dir"], result_dir_name, "iscell.npy")
        np.save(p, iscell)
        job.log(f"import_curation: {iscell.sum()}/{len(iscell)} cells -> {p}", 1)
    return iscell
