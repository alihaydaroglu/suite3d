import numpy as n
import os
import re
import tifffile
from matplotlib import pyplot as plt

try:
    import mrcfile
except:
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
