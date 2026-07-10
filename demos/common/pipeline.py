"""Shared plumbing for the Suite3D demos: data lookup, CLI, pipeline, viewer.

The demos assume you have downloaded the figshare archive. Its layout is::

    <data-root>/
        v1/raw/*.tif
        hippocampus/raw/*.tif
        lbm/raw/*.tif
        manifest.json
        README.md

Nothing here reaches outside `--data-root` and `--out-dir`, so a demo run is
reproducible on any machine.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from suite3d.job import Job
from suite3d import io

from .datasets import DATASETS, get_params


STAGES = ["init", "register", "corrmap", "segment", "extract"]


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ---------------------------------------------------------------------------
# Data location
# ---------------------------------------------------------------------------

def find_tifs(data_root, name):
    """Locate the raw tifs for one dataset inside the figshare download.

    Accepts either the archive root (containing `v1/`, `lbm/`, ...) or the
    dataset folder itself, so `--data-root ./v1` also works.
    """
    folder = DATASETS[name]["folder"]
    root = Path(data_root).expanduser().resolve()

    candidates = [root / folder / "raw", root / "raw", root]
    for cand in candidates:
        if cand.is_dir():
            tifs = io.get_tif_paths(str(cand))
            if tifs:
                return tifs

    raise FileNotFoundError(
        "No .tif files found for dataset %r under %s.\n"
        "Expected the figshare layout <data-root>/%s/raw/*.tif .\n"
        "Looked in:\n  %s\n"
        "Download the archive first; the demos do not fetch data themselves."
        % (name, root, folder, "\n  ".join(str(c) for c in candidates))
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser(name, description):
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-root", type=str, default=".",
                   help="figshare download root (holds %s/raw/*.tif)"
                        % DATASETS[name]["folder"])
    p.add_argument("--out-dir", type=str, required=True,
                   help="where the suite3d job directory is written")
    p.add_argument("--job-id", type=str, default=DATASETS[name]["job_id"],
                   help="name of this run (default: %(default)s)")
    p.add_argument("--viewer", choices=["html", "napari", "none"], default="html",
                   help="what to open when the run finishes (default: %(default)s)")
    p.add_argument("--n-frames", type=int, default=None,
                   help="clip the movie to this many volumes (default: all)")
    p.add_argument("--extract-batch", type=int, default=None,
                   help="volumes held in RAM at once during trace extraction. "
                        "This sets the pipeline's peak memory (default: auto)")
    p.add_argument("--extract-batch-gb", type=float, default=4.0,
                   help="target size of one extraction batch, GiB (default: %(default)s)")
    for s in STAGES:
        p.add_argument("--skip-%s" % s, action="store_true",
                       help="skip the %s stage (re-uses what is on disk)" % s)
    p.add_argument("--overwrite", action="store_true",
                   help="recreate the job directory from scratch")
    return p


def load_or_create_job(args, name):
    """Load an existing job directory, or create one with this dataset's params."""
    out_dir = Path(args.out_dir).expanduser().resolve()
    job_dir = out_dir / ("s3d-%s" % args.job_id)

    if job_dir.exists() and (job_dir / "params.npy").exists() and not args.overwrite:
        log("Loading existing job at %s" % job_dir)
        return Job(str(out_dir), args.job_id, create=False)

    params = get_params(name)
    tifs = find_tifs(args.data_root, name)
    log("Found %d tifs for %r" % (len(tifs), name))
    log("Volume rate fs=%.4f Hz over %d planes"
        % (params["fs"], len(params["planes"])))

    return Job(str(out_dir), args.job_id, tifs=tifs, params=params,
               create=True, overwrite=True, verbosity=3)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_stages(job, args):
    """Run init -> register -> corrmap -> segment -> npil+extract."""
    if not args.skip_init:
        log("=== 1/5 initialization ===")
        job.run_init_pass()

    if not args.skip_register:
        log("=== 2/5 registration ===")
        job.register()

    if not args.skip_corrmap:
        log("=== 3/5 correlation map ===")
        if args.n_frames is None:
            job.calculate_corr_map()
        else:
            # Clip the registered movie so the corrmap spans a fixed duration.
            # The clip happens here, not via a parameter: registration always
            # reads whole tifs.
            mov = job.get_registered_movie()          # (nz, nt, ny, nx)
            nf = min(args.n_frames, mov.shape[1])
            if nf < args.n_frames:
                log("WARNING: only %d volumes registered, wanted %d"
                    % (mov.shape[1], args.n_frames))
            log("corrmap on %d of %d volumes" % (nf, mov.shape[1]))
            job.calculate_corr_map(mov=mov[:, :nf])

    if not args.skip_segment:
        log("=== 4/5 segmentation ===")
        job.segment_rois()

    if not args.skip_extract:
        log("=== 5/5 neuropil masks + trace extraction + deconvolution ===")
        job.compute_npil_masks()
        job.extract_and_deconvolve(batchsize_frames=extract_batch(job, args))

    return job


def extract_batch(job, args):
    """How many volumes to pull into RAM at once during trace extraction.

    This is what sets the pipeline's peak memory. Extraction loads
    `batchsize_frames` volumes of the *registered* movie — which is float32 —
    and `create_shmem_from_arr(..., copy=True)` then duplicates it, so the
    transient cost is roughly `2 x n_z x batch x n_y x n_x x 4 bytes`.

    Suite3D's default of 500 is fine for a 7-plane cortical FOV (~2 GiB) and
    catastrophic for a 22-plane LBM volume (28 GiB per batch, ~56 GiB with the
    copy). Batches are independent — `F[:, start:end]` depends only on its own
    batch — so changing this moves peak RAM and nothing else. Verified on demo
    02: `Fneu`/`spks` come out bit-identical, `F` to 7e-9 (float32 summation
    order).

    **Snap to the registered movie's chunk size.** It is stored in fixed-size
    blocks on disk (100 volumes), so a batch that is not a whole multiple of a
    chunk still forces dask to read whole chunks — and a batch that straddles a
    boundary reads *two*. Measured on demo 02 (22 planes), extraction alone:

        batch 500 (5 chunks) : 56 GiB of batch  -> 113.5 GiB whole-run peak
        batch  35 (straddles):  24.4 GiB peak, 9m40s   <- smaller batch, WORSE
        batch 100 (1 chunk)  :  22.4 GiB peak, 6m36s   <- both cheaper and faster

    So we pick the largest whole number of chunks that fits the budget, and
    never go below one chunk.
    """
    if args.extract_batch:
        return args.extract_batch

    mov = job.get_registered_movie()
    nz, nt, ny, nx = mov.shape
    chunk = mov.chunks[1][0]                    # volumes per on-disk block
    per_frame = nz * ny * nx * mov.dtype.itemsize * 2   # x2 for the shmem copy

    n_chunks = int(args.extract_batch_gb * (1024 ** 3) / (per_frame * chunk))
    batch = min(max(1, n_chunks) * chunk, nt)
    log("extraction batch: %d volumes = %d x %d-volume chunk (~%.1f GiB incl. "
        "the shmem copy)" % (batch, batch // chunk, chunk,
                             batch * per_frame / 1024 ** 3))
    return batch


def n_rois(job):
    """Number of detected ROIs, or None if segmentation has not run."""
    try:
        # to_load with a single entry returns the array itself, not a dict.
        return len(job.load_segmentation_results(to_load=["stats"]))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

def open_napari(job):
    """Open the 3D napari viewer on a finished job.

    `create_napari_ui` lives in `suite3d.ui`, not `suite3d.curation`, and takes
    an `outputs` dict (stats/iscell/vmap), not a Job.
    """
    from suite3d.ui import create_napari_ui
    import napari

    outputs = job.load_segmentation_results()          # info, stats, iscell
    outputs.update(job.load_corr_map_results())        # max_img, mean_img, vmap

    create_napari_ui(outputs, scale=job.params["voxel_size_um"])
    napari.run()


def open_viewer(job, mode, export_dir):
    """Finish a run by exporting results and opening the requested viewer."""
    if mode == "none":
        log("Skipping viewer (--viewer none). Results are in the job directory.")
        return

    if mode == "napari":
        log("Opening the napari UI ...")
        open_napari(job)
        return

    # mode == "html". export_results() writes <export_dir>/s3d-results-<job_id>/
    if hasattr(job, "make_html_viewer"):
        job.export_results(str(export_dir), make_viewer=True)
        log("Open %s/s3d-results-%s/viewer.html in a browser. The directory is "
            "portable — copy it anywhere." % (export_dir, job.job_id))
    else:
        # job.make_html_viewer() is specced but not implemented in this build;
        # export anyway so the run stays useful rather than failing at the end.
        job.export_results(str(export_dir))
        log("Results exported to %s/s3d-results-%s" % (export_dir, job.job_id))
        log("NOTE: this build of suite3d has no job.make_html_viewer(), so no "
            "viewer.html was written. Use --viewer napari, or upgrade suite3d.")


def main(name, description):
    """Entry point shared by every dataset demo."""
    args = build_parser(name, description).parse_args()
    job = load_or_create_job(args, name)
    run_stages(job, args)

    count = n_rois(job)
    if count is not None:
        expected = DATASETS[name].get("expected_rois")
        suffix = " (reference run: %d)" % expected if expected else ""
        log("Detected %d ROIs.%s" % (count, suffix))

    open_viewer(job, args.viewer, export_dir=Path(args.out_dir).expanduser().resolve())
    log("Done.")
    return job
