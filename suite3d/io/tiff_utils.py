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


def show_tif_all_planes(
    img,
    figsize=(8, 6),
    title=None,
    suptitle=None,
    ncols=5,
    same_scale=False,
    vminmax_percentile=(0.5, 99.5),
    vminmax=None,
    **kwargs,
):
    """
    Uses show_tif to create a figure which shows all planes

    Parameters
    ----------
    img : ndarray (nz, ny, nx)
        A 3D image, will show each plane seperatley
    figsize : tuple, optional
        figsize, best if it is a multiple of (ncols, nrows), by default (8,6)
    title : list, optional
        A list of title for each image, by default None
    ncols : int, optional
        The number of collumns in the image, by default 5
    same_scale : bool, optional
        If True enforce all images to have the same colour scale, by default False
    vminvmax_percentile : tuple, optional
        Same as in show_tiff, however isused in getting the same scale if same_scale = True, by default (0.5, 99.5)
    """
    nz = img.shape[0]
    ncols = ncols
    nrows = int(n.ceil(nz / ncols))

    figsize = figsize  # ideally multiple of rows and colloumns
    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize), layout="constrained")
    if ncols == 1 or nrows == 1:
        if nrows == 1 and ncols == 1:
            axs = [[axs]]
        else:
            axs = [axs]
    # make all the images have the same color scale
    if same_scale:
        if vminmax is None:
            todo("whats going on here?")
            for z in range(nz):
                if z == 0:
                    non_nan = ~n.isnan(img[z])
                    vmin = n.percentile(img[z][non_nan], vminmax_percentile[0])
                    vmax = n.percentile(img[z][non_nan], vminmax_percentile[1])
                else:
                    non_nan = ~n.isnan(img[z])
                    vmin = min(vmin, n.percentile(img[z][non_nan], vminmax_percentile[0]))
                    vmax = max(vmax, n.percentile(img[z][non_nan], vminmax_percentile[1]))
            vminmax = (vmin, vmax)
    for row in range(nrows):
        for col in range(ncols):
            plane_no = row * ncols + col
            if plane_no < nz:  # catch empty planes
                if same_scale:
                    show_tif(img[plane_no], ax=axs[row][col], vminmax=vminmax, **kwargs)
                else:
                    show_tif(
                        img[plane_no],
                        ax=axs[row][col],
                        vminmax_percentile=vminmax_percentile,
                        vminmax=vminmax,
                        **kwargs,
                    )
                if title is None:
                    axs[row][col].set_title(
                        f"Plane {plane_no + 1}", fontsize="small"
                    )  # Counting from 0
                else:
                    axs[row][col].set_title(title[plane_no])
            else:
                # hide axis for empty planes
                axs[row][col].set_axis_off()

    if suptitle is not None:
        fig.suptitle(suptitle)
    return fig, axs


def show_tif(
    im,
    flip=1,
    cmap="Greys_r",
    colorbar=False,
    other_args={},
    figsize=(8, 6),
    dpi=150,
    alpha=None,
    return_fig=True,
    ticks=False,
    ax=None,
    px_py=None,
    exact_pixels=False,
    vminmax_percentile=(0.5, 99.5),
    vminmax=None,
    facecolor="white",
    xticks=None,
    yticks=None,
    norm=None,
    cbar=False,
    cbar_loc="left",
    cbar_fontcolor="k",
    cbar_ori="vertical",
    cbar_title="",
    interpolation="nearest",
    ax_off=False,
    cax_kwargs={"frameon": False},
):

    f = None
    im = im.copy()
    if exact_pixels:
        ny, nx = im.shape
        figsize = (nx / dpi, ny / dpi)
        px_py = None

    new_args = {}
    new_args.update(other_args)
    if ax is None:
        f, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if facecolor is not None:
        ax.set_facecolor(facecolor)
    ax.grid(False)
    new_args["interpolation"] = interpolation
    if vminmax_percentile is not None and vminmax is None:
        non_nan = ~n.isnan(im)
        new_args["vmin"] = n.percentile(im[non_nan], vminmax_percentile[0])
        new_args["vmax"] = n.percentile(im[non_nan], vminmax_percentile[1])
    if vminmax is not None:
        new_args["vmin"] = vminmax[0]
        new_args["vmax"] = vminmax[1]
    if px_py is not None:
        new_args["aspect"] = px_py[1] / px_py[0]
    if alpha is not None:
        if type(alpha) == float:
            alpha = n.ones_like(im) * alpha
        new_args["alpha"] = alpha.copy()
    if norm is not None:
        new_args["norm"] = norm
        new_args["vmin"] = None
        new_args["vmax"] = None
    # print(new_args)
    axim = ax.imshow(flip * im, cmap=cmap, **new_args)
    if colorbar:
        plt.colorbar()
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if exact_pixels:
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # plt.tight_layout()
    if norm:
        new_args["vmin"] = norm.vmin
        new_args["vmax"] = norm.vmax
    if cbar:
        if cbar_loc == "left":
            cbar_loc = [0.025, 0.4, 0.02, 0.2]
            cbar_ori = "vertical"
        elif cbar_loc == "right":
            cbar_loc = [0.88, 0.4, 0.02, 0.2]
            cbar_ori = "vertical"
        elif cbar_loc == "top":
            cbar_loc = [0.4, 0.95, 0.2, 0.02]
            cbar_ori = "horizontal"
        elif cbar_loc == "bottom":
            cbar_loc = [0.4, 0.05, 0.2, 0.02]
            cbar_ori = "horizontal"
        cax = ax.inset_axes(cbar_loc, **cax_kwargs)
        plt.colorbar(axim, cax=cax, orientation=cbar_ori)
        if cbar_ori == "vertical":
            cax.set_yticks(
                [new_args["vmin"], new_args["vmax"]],
                ["%.2f" % new_args["vmin"], "%.2f" % new_args["vmax"]],
                color=cbar_fontcolor,
                fontsize=9,
            )
            cax.set_ylabel(cbar_title, color=cbar_fontcolor, fontsize=9, labelpad=-13)
        if cbar_ori == "horizontal":
            cax.set_xticks(
                [new_args["vmin"], new_args["vmax"]],
                ["%.2f" % new_args["vmin"], "%.2f" % new_args["vmax"]],
                color=cbar_fontcolor,
                fontsize=9,
            )
            cax.set_xlabel(cbar_title, color=cbar_fontcolor, fontsize=9, labelpad=-13)
    if xticks is not None:
        ax.set_xticks(range(len(xticks)), xticks)
    if yticks is not None:
        ax.set_yticks(range(len(yticks)), yticks)
    if ax_off:
        ax.axis("off")

    if return_fig:
        return f, ax, axim


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
        mrc.set_data(data.astype(dtype))
        mrc.voxel_size = voxel_size
