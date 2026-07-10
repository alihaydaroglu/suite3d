"""Movie snippets for the portable HTML viewer.

A viewer directory must stay copyable, so a full movie can never go in it: the
registered movie for SS004 is 3 GB per 100 volumes.  What ships instead is a
*snippet* -- a bounded number of frames, contrast-scaled and colour-mapped here in
Python, written as one small image per (movie, plane, frame).

Two constraints drove the format:

*   The page runs from `file://`, where `getImageData` on a local image taints the
    canvas and throws.  So frames are `<img>` elements composited in CSS, never
    read back into a canvas.  (Same rule as the plane backgrounds.)
*   One image per frame, not a sprite sheet.  A sheet of 100 SS004 planes is
    9280x7350 px -- inside Chrome's dimension limit but 273 MB of decoded RGBA,
    and the browser cannot evict part of it.  Separate frames let the image cache
    do its job, and there is no request-latency cost on `file://`.

Grayscale JPEG, so a frame is one channel.  The pixel budget (`max_pixels`) picks
a spatial downsample factor, and then trims `n_frames` if that is still not
enough; whatever was dropped is logged rather than silently applied.

Do not reach for a lower `quality` to shrink the output: two-photon frames are
photon-noise dominated, and JPEG cannot compress noise.  Measured on TC040, one
512x518 frame is 132 kB at q85 and still 85 kB at q60 -- a visible loss for a 35%
saving.  `n_frames` (linear) and `downsample` (quadratic) are the real knobs.

Measured snippet sizes at the defaults: hippocampus (4 planes, 514x512) 48 MB,
V1 TC040 (7 planes, 512x518) 114 MB, SS004 LBM (22 planes, 928x735) 233 MB after
the budget forces a 2x downsample.  Lower `n_frames` if that is too much.
"""

import os

import numpy as np

# Written as (movie key in meta.js, subdir name).  `mov_sub` is the movie the
# correlation map is computed from -- neuropil-subtracted and time-binned.
REGISTERED = "registered"
DETECTION = "mov_sub"

DEFAULT_SPEC = dict(
    n_frames=60,
    start=0,
    movies=(REGISTERED, DETECTION),
    planes=None,          # None -> every plane; else a list of z indices
    downsample=None,      # None -> chosen from max_pixels
    fmt="jpg",
    quality=85,
    # Sized so a ~512x512 volume of <=8 planes stays at native resolution; SS004
    # (22 x 928 x 735) still needs 3x. Two-photon frames are photon-noise dominated
    # and so compress badly: budget ~0.3 bytes/px of jpeg, not the usual ~0.05.
    max_pixels=6.0e8,
    pct=(1.0, 99.5),
)


def frame_filename(z, t, ext):
    """`viewer/movies/<movie>/` file name.  MIRRORED IN JS (`frameURL` in
    assets/viewer.js); `test_viewer_js_movies` diffs the two."""
    return f"z{z:02d}_t{t:04d}.{ext}"


def normalize_spec(movie_snippet):
    """`True` / `int` / `dict` -> a full spec dict."""
    spec = dict(DEFAULT_SPEC)
    if movie_snippet is True:
        pass
    elif isinstance(movie_snippet, (int, np.integer)) and not isinstance(movie_snippet, bool):
        spec["n_frames"] = int(movie_snippet)
    elif isinstance(movie_snippet, dict):
        bad = set(movie_snippet) - set(DEFAULT_SPEC)
        if bad:
            raise ValueError(f"unknown movie_snippet keys {sorted(bad)}; "
                             f"valid keys are {sorted(DEFAULT_SPEC)}")
        spec.update(movie_snippet)
    else:
        raise TypeError("movie_snippet must be True, an int (n_frames), or a dict")

    if spec["fmt"] not in ("jpg", "png"):
        raise ValueError(f"fmt must be 'jpg' or 'png', got {spec['fmt']!r}")
    if spec["n_frames"] < 1:
        raise ValueError("n_frames must be >= 1")
    if spec["planes"] is not None:
        spec["planes"] = [int(z) for z in spec["planes"]]
        if not spec["planes"]:
            raise ValueError("planes must be None or a non-empty list of z indices")
    return spec


def resolve_planes(spec, nz):
    """The z indices to write, validated against the volume."""
    if spec["planes"] is None:
        return list(range(nz))
    bad = [z for z in spec["planes"] if not 0 <= z < nz]
    if bad:
        raise ValueError(f"planes {bad} out of range for a {nz}-plane volume")
    return sorted(set(spec["planes"]))


def _budget(spec, n_movies, nz, ny, nx):
    """Pick (downsample, n_frames) that fit `max_pixels`.

    Downsampling is preferred over dropping frames: a coarse movie that plays is
    more use than a sharp one that stops after 12 frames.
    """
    n_frames = int(spec["n_frames"])
    max_px = float(spec["max_pixels"])
    per_frame = ny * nx

    if spec["downsample"] is not None:
        ds = max(1, int(spec["downsample"]))
    else:
        ds = 1
        while ds < 8 and n_movies * nz * n_frames * per_frame / (ds * ds) > max_px:
            ds += 1

    fit = int(max_px // (n_movies * nz * per_frame / (ds * ds)))
    return ds, max(1, min(n_frames, fit))


def _downsample(a, ds):
    """Block-mean by `ds` on the last two axes, cropping the ragged edge."""
    if ds == 1:
        return a
    nt, ny, nx = a.shape
    ny2, nx2 = (ny // ds) * ds, (nx // ds) * ds
    a = a[:, :ny2, :nx2]
    return a.reshape(nt, ny2 // ds, ds, nx2 // ds, ds).mean(axis=(2, 4))


def _plane_frames(mov, key, z, t0, n_frames):
    """(n_frames, ny, nx) float32 for one plane.

    THE AXIS ORDER DIFFERS BETWEEN THE TWO MOVIES and getting it wrong yields a
    plausible-looking movie of the wrong thing:
        registered  (nz, nt, ny, nx)   -- from get_registered_movie()
        mov_sub     (nt, nz, ny, nx)   -- time first, already time-binned
    """
    sl = (slice(z, z + 1), slice(t0, t0 + n_frames)) if key == REGISTERED \
        else (slice(t0, t0 + n_frames), slice(z, z + 1))
    a = np.asarray(mov[sl], dtype=np.float32)
    return a[0] if key == REGISTERED else a[:, 0]


def _open_movies(job, spec):
    """Load whichever requested movies actually exist on disk.

    `mov_sub` is routinely deleted after the correlation map is built
    (`run_detection(delete_mov_sub=True)`), so a missing one is normal, not an error.
    """
    out = {}
    for key in spec["movies"]:
        if key == REGISTERED:
            try:
                mov = job.get_registered_movie()
            except Exception as e:                       # noqa: BLE001
                job.log(f"  movies: no registered movie ({e})", 2)
                continue
            if mov is not None:
                out[key] = mov
        elif key == DETECTION:
            if not os.path.isdir(os.path.join(job.dirs["job_dir"], "mov_sub")):
                job.log("  movies: no mov_sub/ (deleted after detection?), skipping", 2)
                continue
            try:
                out[key] = job.get_subtracted_movie()
            except Exception as e:                       # noqa: BLE001
                job.log(f"  movies: could not open mov_sub ({e})", 2)
        else:
            raise ValueError(f"unknown movie {key!r}; expected "
                             f"{REGISTERED!r} or {DETECTION!r}")
    return out


def detection_stride(job, movs, nt_trace=0):
    """How many volumes go into one `mov_sub` bin.

    Derived from the data, not from `params["detection_timebin"]`, for two reasons:
    it is usually `None` on disk (the pipeline fills it in at run time and does not
    always persist it), and when it *is* filled in it comes from `2*round(fs/tau)` --
    which is wrong wherever `params['fs']` holds the per-plane rate rather than the
    volume rate (e.g. s3d-TC040_2025-10-28: the formula says 46, the movie says 3).

    `mov_sub` is written one batch at a time and each batch drops its ragged tail
    (800 volumes at bin 3 -> 266 bins, not 266.67), so the first chunk gives the bin
    size exactly.  The dropped tails also mean bin index -> volume index drifts by a
    couple of frames per batch; the viewer uses this only to place a time cursor.
    """
    sub = movs[DETECTION]
    chunks = getattr(sub, "chunks", None)
    bins0 = chunks[0][0] if chunks else sub.shape[0]
    nt_full = movs[REGISTERED].shape[1] if REGISTERED in movs else nt_trace
    batch = int(job.params.get("t_batch_size", 0) or 0)
    covered = min(batch, nt_full) if (batch and nt_full) else nt_full
    if covered and bins0:
        return max(1, int(round(covered / bins0)))
    return max(1, int(job.params.get("detection_timebin") or 1))


def write_movie_snippets(job, vdir, spec, nz, ny, nx, nt_trace=0):
    """Write `viewer/movies/<movie>/z<NN>_t<NNNN>.<ext>`; return the meta dict.

    Returns {} when no movie is available, which the page treats as "no movie
    controls" rather than an error.
    """
    from PIL import Image                                 # ships with matplotlib

    movs = _open_movies(job, spec)
    if not movs:
        return {}

    planes = resolve_planes(spec, nz)
    ds, n_frames = _budget(spec, len(movs), len(planes), ny, nx)
    if ds > 1 or n_frames != spec["n_frames"]:
        job.log(f"  movies: pixel budget {spec['max_pixels']:.0f} -> "
                f"downsample {ds}x, {n_frames}/{spec['n_frames']} frames", 2)

    timebin = detection_stride(job, movs, nt_trace) if DETECTION in movs else 1
    ext = spec["fmt"]
    lo_pct, hi_pct = spec["pct"]
    meta, total_bytes = {}, 0

    for key, mov in movs.items():
        # `start` is in volumes; mov_sub is indexed in bins of `detection_timebin`.
        t0 = int(spec["start"]) // timebin if key == DETECTION else int(spec["start"])
        stride = timebin if key == DETECTION else 1
        # Slide the window back rather than truncating it: asking for 100 frames
        # from t=10 of a 50-frame movie should give 50, not 40.
        nt_avail = mov.shape[0] if key == DETECTION else mov.shape[1]
        t0 = max(0, min(t0, nt_avail - n_frames))
        nf = int(min(n_frames, nt_avail - t0))
        if nf < 1:
            job.log(f"  movies: {key} has no frames at start={spec['start']}", 2)
            continue

        # The page stretches each frame onto the ROI overlay, so a movie whose FOV
        # differs from the correlation map would show ROIs over the wrong pixels.
        if tuple(mov.shape[-2:]) != (ny, nx):
            job.log(f"  movies: {key} is {mov.shape[-2:]} but the ROI grid is "
                    f"{(ny, nx)}; skipping (frames would not align)", 0)
            continue

        mdir = os.path.join(vdir, "movies", key)
        os.makedirs(mdir, exist_ok=True)

        for z in planes:
            a = _downsample(_plane_frames(mov, key, z, t0, nf), ds)
            fin = a[np.isfinite(a)]
            lo, hi = (np.percentile(fin, [lo_pct, hi_pct]) if fin.size
                      else (0.0, 1.0))
            if hi <= lo:
                hi = lo + 1e-6
            # one scaling for the whole plane snippet, so playback does not flicker
            u8 = np.clip((np.nan_to_num(a) - lo) / (hi - lo), 0, 1)
            u8 = np.round(u8 * 255).astype(np.uint8)
            for t in range(nf):
                p = os.path.join(mdir, frame_filename(z, t, ext))
                im = Image.fromarray(u8[t], mode="L")
                if ext == "jpg":
                    im.save(p, "JPEG", quality=int(spec["quality"]), optimize=True)
                else:
                    im.save(p, "PNG", optimize=True)
                total_bytes += os.path.getsize(p)

        meta[key] = dict(
            label="motion-corrected" if key == REGISTERED else "detection (mov_sub)",
            n_frames=nf, ext=ext, downsample=int(ds),
            planes=planes,                             # the page must not request others
            t0=int(t0 * stride), stride=int(stride),   # -> index into the trace time axis
        )
        job.log(f"  movies: {key} {nf} frames x {len(planes)} planes", 2)

    if meta:
        job.log(f"  movies: {total_bytes/1e6:.1f} MB total", 2)
    return meta
