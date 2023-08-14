import mrcfile
import numpy as n
import os
import psutil
from skimage import io as skio
from multiprocessing import Pool
import time
from scipy import signal
import re
import tifffile
import dask.array as darr
from itertools import product
from suite2p.io import lbm as lbmio
import json
from matplotlib import pyplot as plt

def show_tif(im, flip=1, cmap='Greys_r', colorbar=False, other_args = {},figsize=(8,6), dpi=150, alpha=None, return_fig=True,
             ticks=False, ax = None, px_py=None, exact_pixels=False, vminmax_percentile = (0.5,99.5), vminmax = None, facecolor='white', xticks=None, yticks = None,
             norm=None, cbar=False, cbar_loc='left', cbar_ori='vertical', cbar_title='', interpolation='nearest', ax_off=False, cax_kwargs={'frameon':False}):

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
            cax.set_yticks([new_args['vmin'], new_args['vmax']],['%.2f'% new_args['vmin'], '%.2f' % new_args['vmax']], color='k', fontsize=9)
            cax.set_ylabel(cbar_title, color='k', fontsize=9,labelpad=-13)
        if cbar_ori == 'horizontal':
            cax.set_xticks([new_args['vmin'], new_args['vmax']],['%.2f' % new_args['vmin'], '%.2f' % new_args['vmax']], color='k', fontsize=9)
            cax.set_xlabel(cbar_title, color='k', fontsize=9,labelpad=-13)
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

def get_meso_rois(tif_path, max_roi_width_pix=145):
    tf = tifffile.TiffFile(tif_path)
    artists_json = tf.pages[0].tags["Artist"].value

    si_rois = json.loads(artists_json)['RoiGroups']['imagingRoiGroup']['rois']

    rois = []
    warned = False
    for roi in si_rois:
        if type(roi['scanfields']) != list:
            scanfield = roi['scanfields']
        else: 
            scanfield = roi['scanfields'][n.where(n.array(roi['zs'])==0)[0][0]]

    #     print(scanfield)
        roi_dict = {}
        roi_dict['uid'] = scanfield['roiUuid']
        roi_dict['center'] = n.array(scanfield['centerXY'])
        roi_dict['sizeXY'] = n.array(scanfield['sizeXY'])
        roi_dict['pixXY'] = n.array(scanfield['pixelResolutionXY'])
        if roi_dict['pixXY'][0] > max_roi_width_pix and not warned:
            print("SI ROI pix count in x is %d, which is impossible, setting it to %d" % (roi_dict['pixXY'][0],max_roi_width_pix))
            warned=True
            roi_dict['pixXY'][0] = max_roi_width_pix
    #         print(scanfield)
        rois.append(roi_dict)
    #     print(len(roi['scanfields']))

    roi_pixs = n.array([r['pixXY'] for r in rois])
    return rois

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


def save_mrc(dir, fname, data, voxel_size, dtype=n.float32):
    os.makedirs(dir, exist_ok=True)
    fpath = os.path.join(dir, fname)
    with mrcfile.new(fpath, overwrite=True) as mrc:
        mrc.set_data(data.astype(dtype))
        mrc.voxel_size = voxel_size