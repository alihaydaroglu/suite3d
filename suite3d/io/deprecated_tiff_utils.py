import mrcfile
import numpy as n
import os
import psutil
import dask.array as darr
from itertools import product
from skimage import io as skio
from multiprocessing import Pool
import time
from scipy import signal
from suite2p.io import lbm as lbmio
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from .tiff_utils import show_tif

from ..utils import deprecated


@deprecated("Not used anywhere")
def show_tif_all_planes(img, figsize = (8,6), title = None, suptitle = None, ncols = 5, same_scale = False, vminmax_percentile = (0.5,99.5), **kwargs):
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

    figsize = figsize #ideally multiple of rows and colloumns

    fig, axs =  plt.subplots(nrows, ncols, figsize = (figsize), layout = 'constrained')

    #make all the images have the same color scale
    if same_scale:
        for z in range(nz):
            if z ==0:
                non_nan = ~n.isnan(img[z])
                vmin = n.percentile(img[z][non_nan], vminmax_percentile[0])
                vmax = n.percentile(img[z][non_nan], vminmax_percentile[1])
            else:
                non_nan = ~n.isnan(img[z])
                vmin = min(vmin, n.percentile(img[z][non_nan], vminmax_percentile[0]))
                vmax = max(vmax, n.percentile(img[z][non_nan], vminmax_percentile[1]))


    for row in range(nrows):
        for col in range(ncols):
            plane_no = row * ncols + col
            if plane_no < nz: #catch empty planes
                if same_scale:
                    show_tif(img[plane_no], ax=axs[row][col], vminmax = (vmin, vmax), **kwargs)
                else:
                    show_tif(img[plane_no], ax=axs[row][col], vminmax_percentile = vminmax_percentile, **kwargs)
                if title is None:
                    axs[row][col].set_title(f'Plane {plane_no + 1}', fontsize = 'small')#Counting from 0
                else:
                    axs[row][col].set_title(title[plane_no])
            else:
                #hide axis for empty planes
                axs[row][col].set_axis_off()

    if suptitle is not None:
        fig.suptitle(suptitle)


@deprecated("Not used anywhere, and it doesn't include the planes argument in the call to load_and_stitch_tifs")
def separate_planes_and_save(save_path, tif_paths, channels,
                             ram_fraction=0.5, ram_cap_bytes=None,
                             max_out_file_size_gb=3.0, tifs_per_file=None,
                             n_proc_saving=12, crop=None, dir_per_plane=True,
                             filt=None, filt_params=None, total_n_channels = 30):

    n_planes = len(channels)
    n_tifs = len(tif_paths)
    ram_cap = ram_cap_bytes

    # 1 block = 1 input file
    block_size = os.path.getsize(tif_paths[0])
    block_size_per_plane = block_size / total_n_channels
    block_size_loaded = n_planes * block_size_per_plane

    # empirically it seems like when the process is running it uses
    # 1.5x more ram than the total of the files (for copying etc.)
    extra_ram_factor = 1.5

    if filt_params is not None:
        filt = signal.iirnotch(
            filt_params['f0'], filt_params['Q'], filt_params['line_freq'])

    if tifs_per_file is None:
        free_ram = dict(psutil.virtual_memory()._asdict())['available']
        use_ram = free_ram * (ram_fraction / extra_ram_factor)
        if ram_cap is not None:
            use_ram = max(ram_cap, use_ram)
        print("Capping at %.2f GB of %.2f GB available RAM" %
              (use_ram//(1024**3), free_ram//(1024**3)))

        max_n_blocks_per_file = max_out_file_size_gb * \
            (1024**3) // block_size_per_plane
        max_n_blocks_loaded = use_ram // block_size_loaded

        n_blocks = min(max_n_blocks_loaded, max_n_blocks_per_file)

    else:
        print("Warning: No RAM capping!")
        n_blocks = tifs_per_file

    print("Loading %d files at a time, %d channels, for a total of %.2f GB" %
          (n_blocks, n_planes, n_blocks * block_size_loaded / 1024**3))
    print("Saving %d output files at a time, %.2f GB each for a total of %.2f GB" %
          (n_planes, block_size_per_plane * n_blocks / 1024**3, n_planes * block_size_per_plane * n_blocks / 1024**3, ))

    n_iters = int(n.ceil(n_tifs / n_blocks))

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    tif_idx = 0
    for iter_idx in range(n_iters):
        print("\nIteration %d / %d " % (iter_idx, n_iters))
        tic_load = time.time()
        mov = lbmio.load_and_stitch_tifs(tif_paths[tif_idx:tif_idx+int(n_blocks)], verbose=True, channels=channels,
                                       filt=filt)
        if crop is not None:
            print("Cropping...")
            print(mov.shape)
            y0, y1, x0, x1 = crop
            mov = mov[:, :, y0:y1, x0:x1]
            print(mov.shape)
        toc_load = time.time()
        print("    Loaded %d files in %.2fs" %
              (len(tif_paths[tif_idx:tif_idx+int(n_blocks)]), toc_load-tic_load))

        # PARALLELIZE THIS
        print("    Saving all planes to %s ..." % save_path)
        tic_save = time.time()
        if n_proc_saving == 1:
            for plane_idx in range(n_planes):
                filename = 'plane%02d-%05d.tif' % (plane_idx, iter_idx)
                if dir_per_plane:
                    plane_dir = os.path.join(
                        save_path, 'plane%02d' % plane_idx)
                    os.makedirs(plane_dir, exist_ok=True)
                else:
                    plane_dir = save_path
                filepath = os.path.join(plane_dir, filename)
                skio.imsave(filepath, mov[plane_idx])
        else:
            p = Pool(processes=n_proc_saving)
            if dir_per_plane:
                for plane_idx in range(n_planes):
                    os.makedirs(os.path.join(save_path, 'plane%02d' %
                                plane_idx), exist_ok=True)
                output = p.starmap(save_plane_worker, [(os.path.join(save_path, 'plane%02d' % plane_idx, 'plane%02d-%05d.tif' % (plane_idx, iter_idx)), mov[plane_idx])
                                                       for plane_idx in range(n_planes)])
            else:
                output = p.starmap(save_plane_worker, [(os.path.join(save_path, 'plane%02d-%05d.tif' % (plane_idx, iter_idx)), mov[plane_idx])
                                                       for plane_idx in range(n_planes)])
        toc_save = time.time()
        print("    Saved %d files in %.2fs" % (n_planes, toc_save-tic_save))
        tif_idx += int(n_blocks)

@deprecated("Only used in separate_planes_and_save, which is also deprecated")
def save_plane_worker(filepath, mov):
    debug = True
    tic = time.time()
    if debug:
        print("Saving %s" % filepath)

    skio.imsave(filepath, mov)

    toc = time.time()
    if debug:
        print("Saved in %.2f" % (toc - tic))
    return toc - tic

@deprecated("Only used in old demos")
def save_mrc(dir, fname, data, voxel_size, dtype=n.float32):
    os.makedirs(dir, exist_ok=True)
    fpath = os.path.join(dir, fname)
    with mrcfile.new(fpath, overwrite=True) as mrc:
        mrc.set_data(data.astype(dtype))
        mrc.voxel_size = voxel_size


@deprecated("Only called in animate_gif, which is also deprecated")
def animate_frame(Frame, ax, FrameNo, flip=1, cmap='Greys_r', colorbar=False, alpha=None, other_args = {},
             ticks=False, px_py=None, vminmax_percentile = (0.5,99.5), vminmax = None, facecolor='white', xticks=None, yticks = None,
             norm=None, interpolation='nearest', ax_off=False):
    
    """
    Used to animate a single frame of animate_gif
    """
    im = []
    new_args = {}
    new_args.update(other_args)

    if facecolor is not None:
        ax.set_facecolor(facecolor)
    ax.grid(False)
    new_args['interpolation'] = interpolation
    if vminmax_percentile is not None and vminmax is None:
        non_nan = ~n.isnan(Frame)
        new_args['vmin'] = n.percentile(Frame[non_nan], vminmax_percentile[0])
        new_args['vmax'] = n.percentile(Frame[non_nan], vminmax_percentile[1])
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

    #Add Title as text as artist animation is unique
    title = ax.text(0.5, 1.01, f'Frame {FrameNo}', horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)

    im.extend([ax.imshow(flip*Frame,cmap=cmap, **new_args), title])


    if colorbar: plt.colorbar()
    if not ticks:
        ax.set_xticks([]); ax.set_yticks([]);
    if xticks is not None:
        ax.set_xticks(range(len(xticks)), xticks)
    if yticks is not None:
        ax.set_yticks(range(len(yticks)), yticks)
    if ax_off:
        ax.axis('off')

    return im


@deprecated("Not used anywhere")
def animate_gif(Im3D, SaveDir, interval = 500, repeat_delay = 5000, other_args = {}, figsize=(8,6), dpi=150, exact_pixels=False, vminmax_percentile = (0.5,99.5), vminmax = None,
                  **kwargs):
    """
    This function requires a 3D image e.g (nz, ny, nx) and will return an animated gif.

    Parameters
    ----------
    Im3D : ndarray
        A 3D array, which will be animated over the first axis (0)
    SaveDir : path
        Path to the save directory should end in .gif
    interval : int, optional
        The time delay between frames in ms, by default 500
    repeat_delay : int, optional
        The time delay between repeats of the gif, by default 5000
    other_args : dict, optional
        Optional arguments for the call of show_tif for each plane, by default {}
    figsize : tuple, optional
        figure size, by default (8,6)
    dpi : int, optional
        dpi , by default 150
    exact_pixels : bool, optional
        If true adapt figure size to show the exact pixels, by default False
    vminmax_percentile : tuple, optional
        Threshold to clip movie, by default (0.5,99.5)
    vminmax : tuple, optional
        Values to clip the movie, by default None
    """

    Mov = Im3D.copy()
    nFrames, ny, nx = Mov.shape
    if exact_pixels:
        figsize = (nx / dpi, ny / dpi)
    
    new_args = {}
    new_args.update(other_args)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)


    vminmaxFrame = [0 , 0]
    if vminmax_percentile is not None and vminmax is None:
        non_nan = ~n.isnan(Mov)
        vminmaxFrame[0] = n.percentile(Mov[non_nan], vminmax_percentile[0])
        vminmaxFrame[1] = n.percentile(Mov[non_nan], vminmax_percentile[1])
    if vminmax is not None:
        vminmaxFrame[0] = vminmax[0]
        vminmaxFrame[1] = vminmax[1]

    ims = []
    for i in range(nFrames):
        im = animate_frame(Mov[i], ax, i, vminmax = vminmaxFrame,  **kwargs)
        ims.append(im)

    ani = animation.ArtistAnimation(fig, ims, interval = interval, repeat_delay = repeat_delay)   

    ani.save(SaveDir)

@deprecated("Not used anywhere")
def show_tif_all_planes(img, figsize = (8,6), title = None, suptitle = None, ncols = 5, same_scale = False, vminmax_percentile = (0.5,99.5), **kwargs):
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

    figsize = figsize #ideally multiple of rows and colloumns

    fig, axs =  plt.subplots(nrows, ncols, figsize = (figsize), layout = 'constrained')

    #make all the images have the same color scale
    if same_scale:
        for z in range(nz):
            if z ==0:
                non_nan = ~n.isnan(img[z])
                vmin = n.percentile(img[z][non_nan], vminmax_percentile[0])
                vmax = n.percentile(img[z][non_nan], vminmax_percentile[1])
            else:
                non_nan = ~n.isnan(img[z])
                vmin = min(vmin, n.percentile(img[z][non_nan], vminmax_percentile[0]))
                vmax = max(vmax, n.percentile(img[z][non_nan], vminmax_percentile[1]))


    for row in range(nrows):
        for col in range(ncols):
            plane_no = row * ncols + col
            if plane_no < nz: #catch empty planes
                if same_scale:
                    show_tif(img[plane_no], ax=axs[row][col], vminmax = (vmin, vmax), **kwargs)
                else:
                    show_tif(img[plane_no], ax=axs[row][col], vminmax_percentile = vminmax_percentile, **kwargs)
                if title is None:
                    axs[row][col].set_title(f'Plane {plane_no + 1}', fontsize = 'small')#Counting from 0
                else:
                    axs[row][col].set_title(title[plane_no])
            else:
                #hide axis for empty planes
                axs[row][col].set_axis_off()

    if suptitle is not None:
        fig.suptitle(suptitle)
    

@deprecated("Refactor: use the one in suite3d.utils instead.")
def npy_to_dask(files, name='', axis=1):
    sample_mov = n.load(files[0], mmap_mode='r')
    file_ts = ([n.load(f, mmap_mode='r').shape[axis] for f in files])
    nz, nt_sample, ny, nx = sample_mov.shape

    dtype = sample_mov.dtype
    chunks = [(nz,), (nt_sample,), (ny,), (nx,)]
    chunks[axis] = tuple(file_ts)
    chunks = tuple(chunks)
    name = 'from-npy-stack-%s' % name

    keys = list(product([name], *[range(len(c)) for c in chunks]))
    values = [(n.load, files[i], 'r') for i in range(len(chunks[axis]))]

    dsk = dict(zip(keys, values))

    arr = darr.Array(dsk, name, chunks, dtype)

    return arr