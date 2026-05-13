import numpy as n
import os
import re
import tifffile
from matplotlib import pyplot as plt

try:
    import mrcfile
except ImportError:
    print("No MRCFile")
from .lbmio import get_meso_rois
from ..developer import todo, deprecated
from natsort import natsorted


def get_si_params(tif_path):
    """
    Get scanimage parameters from a tiff file.

    Args:
        tif_path (str): Path to the tiff file.

    Returns:
        dict: Dictionary containing scanimage parameters.
            rois: List of dictionaries containing ROI information.
            vol_rate: Volume rate.
            line_freq: Line frequency.
    """
    todo("Consider integrating in the central s3dio class.")
    si_params = {}
    si_params["rois"] = get_meso_rois(tif_path)
    si_params["vol_rate"] = get_vol_rate(tif_path)
    si_params["line_freq"] = 2 * get_tif_tag(
        tif_path, "SI.hScan2D.scannerFrequency", number=True
    )
    return si_params



def get_tif_paths(dir_path, regex_filter=None, sort=True, natsort=False):
    """
    Get a list of absolute paths for all tif files in this directory

    Args:
        dir_path (str): Directory containing tifs
        regex_filter (string, optional): Optional regex filter for tif names. Defaults to None.

    Returns:
        list: list of tif paths
    """
    dir_path_ls = os.listdir(dir_path)
    tif_paths = [os.path.join(dir_path, e) for e in dir_path_ls if e.endswith(".tif")]
    if regex_filter is not None:
        tif_paths_filtered = []
        for tif_path in tif_paths:
            if re.search(regex_filter, tif_path) is not None:
                tif_paths_filtered.append(tif_path)
        tif_paths = tif_paths_filtered

    if sort:
        if natsort:
            tif_paths = natsorted(tif_paths)
        else:
            tif_paths = sorted(tif_paths)  # list(n.sort(tif_paths))
    return tif_paths


def get_tif_tag(tif_path, tag_name=None, number=True):
    tf = tifffile.TiffFile(tif_path)
    tags = tf.pages[0].tags["Software"].value.split("\n")
    if tag_name is None:
        return tags
    for tag in tags:
        if tag_name in tag:
            if number:
                tag = float(tag.split(" ")[-1])
            return tag


def get_vol_rate(tif_path):
    tf = tifffile.TiffFile(tif_path)
    tags = tf.pages[0].tags["Software"].value.split("\n")
    for tag in tags:
        if tag.startswith("SI.hRoiManager.scanFrameRate"):
            return float(re.findall("\d+\.\d+", tag)[0])


def get_scan_rate(tif_path):
    tf = tifffile.TiffFile(tif_path)
    tif_info = (tf.pages[0].tags["Software"].value).split("\n")
    for line in tif_info:
        if line.startswith("SI.hRoiManager.scanFrameRate"):
            return float(line.split(" ")[-1])


def get_fastZ(tif_path):
    tf = tifffile.TiffFile(tif_path)
    tif_info = (tf.pages[0].tags["Software"].value).split("\n")
    for line in tif_info:
        if line.startswith("SI.hFastZ.position"):
            return float(line.split(" ")[-1])
    return None


def get_frame_counts(tif_paths, safe_mode=False):
    """Measure the number of frames in a list of tif files.
    
    In safe mode, the number of frames is determined by reading the tif files into memory
    and explictly measuring the shape. In unsafe mode (default), the number of frames of
    the first tif is used to calculate a conversion factor from the number of bytes to the
    number of frames and this conversion factor is used to estimate the number of frames in
    each tif, which is much faster but has the potential to fail!
    """
    tif_frames = {}
    if safe_mode:
        for tf in tif_paths:
            tif_frames[tf] = tifffile.imread(tf).shape[0]
    else:
        first_tif_num_frames = tifffile.imread(tif_paths[0]).shape[0]
        bytes_to_frames = float(first_tif_num_frames) / os.path.getsize(tif_paths[0])
        for tf in tif_paths:
            tif_frames[tf] = int(n.round(os.path.getsize(tf) * bytes_to_frames))
    return tif_frames

@deprecated("Only used in old demos")
def save_mrc(dir, fname, data, voxel_size, dtype=n.float32):
    os.makedirs(dir, exist_ok=True)
    fpath = os.path.join(dir, fname)
    with mrcfile.new(fpath, overwrite=True) as mrc:
        print(fpath)
        mrc.set_data(data.astype(dtype))
        mrc.voxel_size = voxel_size


def split_oversized_tiff(
    src_path,
    dst_dir,
    frames_per_chunk=500,
    n_ch_tif=30,
    num_colors=1,
    overwrite=False,
    verbose=True,
):
    """Stream-split a multi-page TIFF into chunks aligned to volume boundaries.

    Use when a single ScanImage 2P TIFF is too large to load into RAM in one
    shot (the suite3d loader calls ``tifffile.imread(path)`` on the whole
    file). The output chunks can be passed to a Job as if they were the
    original list of TIFFs; the pipeline never sees the giant file.

    The split is page-granular: a TIFF "page" is one frame for single-color
    ScanImage data and one frame*color for multi-color data. The chunk size
    is rounded down to a multiple of ``n_ch_tif * num_colors`` so each chunk
    holds whole volumes. Only the final chunk may have a partial trailing
    volume; the existing extra-frames spillover logic in s3dio handles that.

    Uses tifffile.memmap on the source so pages are read on demand rather
    than all at once; safe to call on files larger than RAM.

    Args:
        src_path (str): Absolute path to the source TIFF.
        dst_dir (str): Directory to write chunk files into. Created if needed.
        frames_per_chunk (int): Approximate target pages per chunk. Rounded
            down to a multiple of ``n_ch_tif * num_colors``.
        n_ch_tif (int): Number of planes per volume in the source.
        num_colors (int): Number of color channels per frame.
        overwrite (bool): If False, refuse to overwrite existing chunk files.
        verbose (bool): Print per-chunk progress.

    Returns:
        list[str]: Paths to the chunk files, in order.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)

    pages_per_volume = n_ch_tif * num_colors
    chunk_pages = (frames_per_chunk // pages_per_volume) * pages_per_volume
    if chunk_pages <= 0:
        raise ValueError(
            f"frames_per_chunk={frames_per_chunk} is smaller than one "
            f"volume of {pages_per_volume} pages (n_ch_tif*num_colors). "
            "Increase frames_per_chunk."
        )

    os.makedirs(dst_dir, exist_ok=True)

    # Read page-0 description once for metadata preservation.
    with tifffile.TiffFile(src_path) as tf:
        desc0 = tf.pages[0].description
        n_pages = len(tf.pages)

    # Memmap the source: no full-file read, OS pages on demand.
    src_data = tifffile.memmap(src_path)
    if src_data.shape[0] != n_pages:
        raise RuntimeError(
            f"Page count mismatch: TiffFile reports {n_pages}, memmap "
            f"reports {src_data.shape[0]}. Source TIFF may be unusual; "
            "consider splitting it with an external tool instead."
        )

    src_base = os.path.splitext(os.path.basename(src_path))[0]
    n_chunks = int(n.ceil(n_pages / chunk_pages))
    chunk_paths = []

    if verbose:
        print(
            f"Splitting {src_path}\n"
            f"  pages={n_pages}, chunk_pages={chunk_pages}, "
            f"n_chunks={n_chunks}, pages_per_volume={pages_per_volume}"
        )

    for k in range(n_chunks):
        start = k * chunk_pages
        end = min(start + chunk_pages, n_pages)
        dst_path = os.path.join(dst_dir, f"{src_base}_{k:04d}.tif")
        if os.path.exists(dst_path) and not overwrite:
            raise FileExistsError(
                f"{dst_path} exists. Pass overwrite=True to replace."
            )

        with tifffile.TiffWriter(dst_path, bigtiff=True) as tw:
            tw.write(
                src_data[start:end],
                description=desc0,
                contiguous=True,
            )

        # Sanity-check page count of the written chunk.
        with tifffile.TiffFile(dst_path) as tf_out:
            n_out = len(tf_out.pages)
        if n_out != (end - start):
            raise RuntimeError(
                f"Chunk {dst_path} has {n_out} pages, expected "
                f"{end - start}."
            )

        if verbose:
            print(
                f"  chunk {k+1}/{n_chunks}: pages [{start}, {end}) "
                f"-> {dst_path}"
            )
        chunk_paths.append(dst_path)

    if verbose:
        print(f"Done. Wrote {len(chunk_paths)} chunks to {dst_dir}")
    return chunk_paths


def _split_tiff_cli():
    """CLI entry point for split_oversized_tiff. Wired in pyproject.toml as
    ``s3d-split-tiff``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="s3d-split-tiff",
        description=(
            "Stream-split an oversized ScanImage TIFF into volume-aligned "
            "chunks so suite3d can load it without OOM. See "
            "suite3d/debugging_tips.md 'Oversized TIFFs' for context."
        ),
    )
    parser.add_argument("src", help="Path to the source TIFF.")
    parser.add_argument("dst_dir", help="Output directory for chunk files.")
    parser.add_argument(
        "--frames-per-chunk", type=int, default=500,
        help="Target pages per chunk (default 500). Rounded down to a "
        "multiple of n_ch_tif*num_colors.",
    )
    parser.add_argument(
        "--n-ch-tif", type=int, default=30,
        help="Planes per volume in the source TIFF (default 30).",
    )
    parser.add_argument(
        "--num-colors", type=int, default=1,
        help="Color channels per frame (default 1).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing chunk files in dst_dir.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-chunk progress.",
    )
    args = parser.parse_args()

    split_oversized_tiff(
        src_path=args.src,
        dst_dir=args.dst_dir,
        frames_per_chunk=args.frames_per_chunk,
        n_ch_tif=args.n_ch_tif,
        num_colors=args.num_colors,
        overwrite=args.overwrite,
        verbose=not args.quiet,
    )
