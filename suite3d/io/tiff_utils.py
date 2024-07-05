import numpy as n
import os
import re
import tifffile
from matplotlib import pyplot as plt
import mrcfile
from .lbmio import get_meso_rois
from ..utils import todo, deprecated

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
    todo("Refactor get_meso_rois to combine with the one in lbmio!")
    todo("Consider integrating in the central s3dio class.")
    si_params = {}
    si_params['rois'] = get_meso_rois(tif_path)
    si_params['vol_rate'] = get_vol_rate(tif_path)
    si_params['line_freq'] = 2 * get_tif_tag(tif_path,'SI.hScan2D.scannerFrequency', number=True)
    return si_params

def show_tif(im, flip=1, cmap='Greys_r', colorbar=False, other_args = {},figsize=(8,6), dpi=150, alpha=None, return_fig=True,
             ticks=False, ax = None, px_py=None, exact_pixels=False, vminmax_percentile = (0.5,99.5), vminmax = None, facecolor='white', xticks=None, yticks = None,
             norm=None, cbar=False, cbar_loc='left', cbar_fontcolor = 'k', cbar_ori='vertical', cbar_title='', interpolation='nearest', ax_off=False, cax_kwargs={'frameon':False}):

    f = None
    im = im.copy()
    if exact_pixels:
        ny, nx = im.shape
        figsize = (nx / dpi, ny / dpi)
        px_py = None
    
    new_args = {}
    new_args.update(other_args)
    if ax is None: f,ax = plt.subplots(figsize=figsize, dpi=dpi)

    if facecolor is not None:
        ax.set_facecolor(facecolor)
    ax.grid(False)
    new_args['interpolation'] = interpolation
    if vminmax_percentile is not None and vminmax is None:
        non_nan = ~n.isnan(im)
        new_args['vmin'] = n.percentile(im[non_nan], vminmax_percentile[0])
        new_args['vmax'] = n.percentile(im[non_nan], vminmax_percentile[1])
    if vminmax is not None:
        new_args['vmin'] = vminmax[0]
        new_args['vmax'] = vminmax[1]
    if px_py is not None:
        new_args['aspect'] = px_py[1]/px_py[0]
    if alpha is not None:
        new_args['alpha'] = alpha.copy()
    if norm is not None:
        new_args['norm'] = norm
        new_args['vmin'] = None; new_args['vmax'] = None
    # print(new_args)
    axim = ax.imshow(flip*im,cmap=cmap, **new_args)
    if colorbar: plt.colorbar()
    if not ticks:
        ax.set_xticks([]); ax.set_yticks([]);
    if exact_pixels:
        plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    # plt.tight_layout()
    if norm: 
        new_args['vmin'] = norm.vmin; new_args['vmax'] = norm.vmax
    if cbar:
        if cbar_loc == 'left':
            cbar_loc = [0.025, 0.4, 0.02, 0.2]; cbar_ori='vertical'
        elif cbar_loc == 'right':
            cbar_loc = [0.88, 0.4, 0.02, 0.2]; cbar_ori='vertical'
        elif cbar_loc == 'top':
            cbar_loc = [0.4, 0.95, 0.2, 0.02]; cbar_ori='horizontal'
        elif cbar_loc == 'bottom':
            cbar_loc = [0.4, 0.05, 0.2, 0.02]; cbar_ori='horizontal'
        cax = ax.inset_axes(cbar_loc, **cax_kwargs)
        plt.colorbar(axim, cax=cax, orientation=cbar_ori)
        if cbar_ori == 'vertical':
            cax.set_yticks([new_args['vmin'], new_args['vmax']],['%.2f'% new_args['vmin'], '%.2f' % new_args['vmax']], color=cbar_fontcolor, fontsize=9)
            cax.set_ylabel(cbar_title, color=cbar_fontcolor, fontsize=9,labelpad=-13)
        if cbar_ori == 'horizontal':
            cax.set_xticks([new_args['vmin'], new_args['vmax']],['%.2f' % new_args['vmin'], '%.2f' % new_args['vmax']], color=cbar_fontcolor, fontsize=9)
            cax.set_xlabel(cbar_title, color=cbar_fontcolor, fontsize=9,labelpad=-13)
    if xticks is not None:
        ax.set_xticks(range(len(xticks)), xticks)
    if yticks is not None:
        ax.set_yticks(range(len(yticks)), yticks)
    if ax_off:
        ax.axis('off')

    if return_fig: return f, ax, axim

def get_tif_paths(dir_path, regex_filter=None, sort=True):
    '''
    Get a list of absolute paths for all tif files in this directory

    Args:
        dir_path (str): Directory containing tifs
        regex_filter (string, optional): Optional regex filter for tif names. Defaults to None.

    Returns:
        list: list of tif paths
    '''
    dir_path_ls = os.listdir(dir_path)
    tif_paths = [os.path.join(dir_path, e)
                 for e in dir_path_ls if e.endswith('.tif')]
    if regex_filter is not None:
        tif_paths_filtered = []
        for tif_path in tif_paths:
            if re.search(regex_filter, tif_path) is not None:
                tif_paths_filtered.append(tif_path)
        tif_paths = tif_paths_filtered

    # print(tif_paths)
    if sort: tif_paths = sorted(tif_paths) #list(n.sort(tif_paths))
    return (tif_paths)

def get_tif_tag(tif_path, tag_name=None, number=True):
    tf = tifffile.TiffFile(tif_path)
    tags = tf.pages[0].tags['Software'].value.split('\n')
    if tag_name is None:
        return tags
    for tag in tags:
        if tag_name in tag:
            if number:
                tag = float(tag.split(' ')[-1])
            return tag


def get_vol_rate(tif_path):
    tf = tifffile.TiffFile(tif_path)
    tags = tf.pages[0].tags['Software'].value.split('\n')
    for tag in tags:
        if tag.startswith('SI.hRoiManager.scanFrameRate'):
            return float(re.findall("\d+\.\d+", tag)[0])


def get_scan_rate(tif_path):
    tf = tifffile.TiffFile(tif_path)
    tif_info = (tf.pages[0].tags['Software'].value).split('\n')
    for line in tif_info:
        if line.startswith('SI.hRoiManager.scanFrameRate'):
            return float(line.split(' ')[-1])
        
def get_fastZ(tif_path):
    tf = tifffile.TiffFile(tif_path)
    tif_info = (tf.pages[0].tags['Software'].value).split('\n')
    for line in tif_info:
        if line.startswith('SI.hFastZ.position'):
            return float(line.split(' ')[-1])
    return None

@deprecated("Only used in old demos")
def save_mrc(dir, fname, data, voxel_size, dtype=n.float32):
    os.makedirs(dir, exist_ok=True)
    fpath = os.path.join(dir, fname)
    with mrcfile.new(fpath, overwrite=True) as mrc:
        mrc.set_data(data.astype(dtype))
        mrc.voxel_size = voxel_size


