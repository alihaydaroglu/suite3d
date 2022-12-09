import numpy as n
import os
import matplotlib.pyplot as plt



def show_tif(im, flip=1, cmap='Greys_r', colorbar=False, other_args = {},figsize=(8,6), dpi=150, alpha=None,
             ticks=False, ax = None, px_py=None, exact_pixels=False, vmax_percentile = 99.0, vminmax = None):
    f = None

    if exact_pixels:
        ny, nx = im.shape
        figsize = (nx / dpi, ny / dpi)
        px_py = None

    if ax is None: f,ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.grid(False)
    if 'interpolation' not in other_args.keys():
        other_args['interpolation'] = 'nearest'
    if vmax_percentile is not None and vminmax is None:
        other_args['vmin'] = im.min()
        other_args['vmax'] = n.percentile(im, vmax_percentile)
    if vminmax is not None:
        other_args['vmin'] = vminmax[0]
        other_args['vmax'] = vminmax[1]
    if px_py is not None:
        other_args['aspect'] = px_py[1]/px_py[0]
    if alpha is not None:
        other_args['alpha'] = alpha
    im = ax.imshow(flip*im,cmap=cmap, **other_args)
    if colorbar: plt.colorbar()
    if not ticks:
        ax.set_xticks([]); ax.set_yticks([]);
    if exact_pixels:
        plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    # plt.tight_layout()
    return f, ax, im
