import os
import numpy as n
from multiprocessing import Pool
from scipy.signal import find_peaks
from scipy.stats import gamma
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import imreg_dft as imreg
from multiprocessing import Pool, shared_memory
from suite2p import default_ops
# import tensorflow as tf
# from tensorflow.keras.models import load_model
from itertools import product
from dask import array as darr
from skimage.measure import moments
from suite2p.registration.nonrigid import make_blocks
import pickle

def make_blocks_3d(nz, ny, nx, block_shape, z_overlap=True):
    ybls, xbls,(n_y_bls, n_x_bls), __, __ = make_blocks(ny, nx, block_size=block_shape[1:])
    z_bl_starts = n.arange(0, nz - int(z_overlap), block_shape[0] - int(z_overlap))
    zbls = n.stack([z_bl_starts, z_bl_starts + block_shape[0]],axis=1)
    zbls[zbls > nz] = nz
    n_z_bls = len(zbls)
    grid_shape = (n_z_bls, n_y_bls, n_x_bls)
    ybls = n.concatenate([ybls] * n_z_bls)
    xbls = n.concatenate([xbls] * n_z_bls)
    zbls = n.stack([zbls]*(n_y_bls * n_x_bls),axis=1).reshape(-1,2)
    
    return zbls, ybls, xbls, grid_shape

def get_shifts_3d(im3d, n_procs = 12, filter_pcorr=0):
    sims = []
    i = 0
    p = Pool(n_procs)
    sims = p.starmap(get_shifts_3d_worker, [(idx, im3d, filter_pcorr) for idx in range(im3d.shape[0]-1)])
    tvecs = n.array([sim['tvec'] for sim in sims])
    tvecs_cum = n.cumsum(tvecs,axis=0)
    return tvecs_cum

def get_shifts_3d_worker(idx, im3d,filter_pcorr):
    return imreg.similarity(im3d[idx], im3d[idx+1], filter_pcorr=filter_pcorr)
    
def gaussian(x, mu, sigma):
    return n.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * n.sqrt(2*n.pi))

def sum_log_lik_one_line(m, x, y, b = 0, sigma_0 = 10,  c = 1e-10, m_penalty=0):
    mu = m * x + b
    lik_line = gaussian(y, mu, sigma_0)
    lik = lik_line
    
    log_lik = n.log(lik + c - m * m_penalty).sum()
    
    return -log_lik

def calculate_crosstalk_coeff(im3d, exclude_below=1, sigma=0.01, peak_width=1,     
                            verbose=True, estimate_gamma=True, estimate_from_last_n_planes=None,
                            n_proc = 1, show_plots=True, save_plots = None, force_positive=True,
                            m_penalty = 0, bounds=None, fit_above_percentile=0, fig_scale=3):
    plt.style.use('seaborn')
    m_opts = [] 
    m_firsts = []
    all_liks = []
    m_opt_liks = []
    m_first_liks = []
    im3d = im3d.copy()
    if force_positive:
        im3d = im3d - im3d.min(axis=(1,2),keepdims=True)

    ms = n.linspace(0,1,101)
    assert im3d.shape[0] == 30

    if estimate_from_last_n_planes is None:
        estimate_from_last_n_planes = 15

    if save_plots is not None:
        plot_dir = os.path.join(save_plots, 'crosstalk_plots')
        os.makedirs(plot_dir, exist_ok=True)

    fs = []
    n_plots = estimate_from_last_n_planes
    n_cols = 5
    n_rows = n.ceil(n_plots / n_cols).astype(int)

    # print(n_plots, n_rows, n_cols)
    # print(estimate_from_last_n_planes)
    f,axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*fig_scale, n_rows*fig_scale))
    if n_rows == 1: axs = [axs]
    

    for idx, i in enumerate(range(15 - estimate_from_last_n_planes, 15)):
        # print("Plot for plane %d" % i)
        X = im3d[i].flatten()
        Y = im3d[i+15].flatten()
        fit_thresh = n.percentile(X, fit_above_percentile)
        # print(fit_thresh)
        idxs = X > n.percentile(X, fit_above_percentile)
        # print(len(idxs), X.shape)

        if n_proc == 1:
            liks = n.array([sum_log_lik_one_line(m, X[idxs], Y[idxs], sigma_0 = sigma, m_penalty=m_penalty) for m in ms])
        else:
            p = Pool(n_proc)
            liks = p.starmap(sum_log_lik_one_line,[(m, X[idxs], Y[idxs],0, sigma,1e-10,m_penalty) for m in ms])
            liks = n.array(liks)

        m_opt = ms[n.argmin(liks)]
        pks = find_peaks(-liks, width=peak_width)[0]
        m_first = ms[pks[0]]

        m_opts.append(m_opt)
        m_firsts.append(m_first)
        all_liks.append(liks)
        m_opt_liks.append(liks.min())
        m_first_liks.append(liks[pks[0]])

        if verbose:
            print("Plane %d and %d, m_opt: %.2f and m_first: %.2f" % (i, i+15, m_opt, m_first))
        
    
        if bounds is None: 
            bounds = (0, n.percentile(X,99.95))
        bins = [n.arange(*bounds,1),n.arange(*bounds,1)]
        col_id = idx % n_cols
        row_id = idx // n_cols
        # print(i,idx, col_id, row_id)
        ax = axs[row_id][col_id]
        ax.set_aspect('equal')
        ax.hist2d(X, Y, bins = bins, norm=colors.LogNorm());
        ax.plot(bins[0], m_opt * bins[0], alpha=0.5, linestyle='--')
        ax.plot(bins[0], m_first * bins[0], alpha=0.5, linestyle='--')
        axsins2 = inset_axes(ax, width="30%", height="40%", loc='upper right')
        axsins2.grid(False)
        axsins2.plot(ms, liks, label='Min: %.2f, 1st: %.2f' % (m_opt, m_first))
        # axsins2.set_xlabel("m")
        axsins2.set_xticks([m_opt])
        axsins2.set_yticks([])
        ax.set_xlabel("Plane %d" % i)
        ax.set_ylabel("Plane %d" % (i+15))

    plt.tight_layout()
    # print('showing')
    # if show_plots: plt.show()
    # print("showed")
    print("Saving figure to %s" % plot_dir)
    if save_plots is not None:
        f.savefig(os.path.join(plot_dir, 'plane_fits.png'), dpi=200)
    print("saved")
    plt.close()
    print("Close figure")

    # return
    m_opts = n.array(m_opts)
    m_firsts = n.array(m_firsts)
    
    best_ms = m_opts[m_opts==m_firsts]
    best_m = best_ms.mean()
    
    if estimate_gamma:
        gx = gamma.fit(m_opts)
        x = n.linspace(0,1,1001)
        gs = gamma.pdf(x, *gx)
        f = plt.figure(figsize=(3,3))
        plt.hist(m_opts,density=True, log=False, bins = n.arange(0,1.01, 0.01))
        plt.plot(x,gs)
        plt.yticks([])
        plt.scatter([x[n.argmax(gs)]], [n.max(gs)], label='Best coeff: %.3f' % x[n.argmax(gs)])
        plt.legend()
        plt.xlabel("Coeff value")
        plt.ylabel("")
        plt.xlim(0,0.4)
        plt.title("Histogram of est. coefficients per plane")
        if save_plots is not None:
            plt.savefig(os.path.join(plot_dir, 'gamma_fit.png'), dpi=200)
        if show_plots:
            plt.show()
        plt.close()
        fs.append(f)
        best_m = x[n.argmax(gs)]

    return m_opts, m_firsts, best_m


def shift_movie_plane(plane_id, sh_mem_name, tvec, shape, dtype, verbose=True):
    sh_mem = shared_memory.SharedMemory(sh_mem_name)
    mov3d = n.ndarray(shape, dtype, buffer=sh_mem.buf)
    plane = mov3d[plane_id]
    tvec = tvec
    for i in range(plane.shape[0]):
        # if i % 100 == 0:
            # if verbose:
                # print("Plane %02d: %d " %  (plane_id, i))
        mov3d[plane_id][i] = imreg.transform_img(mov3d[plane_id][i], tvec=tvec)
    sh_mem.close()

def register_movie(mov3d, tvecs = None, save_path = None, n_shift_proc=10):
    
    if tvecs is None:
        im3d = mov3d.mean(axis=1)
        tvecs = get_shifts_3d(im3d, save_path)
    
    n_planes = mov3d.shape[0]
    shape_mem = mov3d.shape
    size_mem = mov3d.nbytes

    sh_mem = shared_memory.SharedMemory(create=True, size=size_mem)

    mov_reg = n.ndarray(shape_mem, dtype=mov3d.dtype, buffer = sh_mem.buf)
    mov_reg[:] = mov3d[:]

    sh_mem_name = sh_mem.name
    p = Pool(n_shift_proc)

    p.starmap(shift_movie_plane, [(idx, sh_mem_name, tvecs[idx], shape_mem, mov_reg.dtype) for idx in n.arange(1,n_planes)])

    im3d = mov_reg.mean(axis=1)
    mov_reg_ret = mov_reg.copy()
    sh_mem.close()
    sh_mem.unlink()

    return mov_reg_ret


def build_ops(save_path, recording_params, other_params):
    ops = default_ops()
    # files
    ops['fast_disk'] = []
    ops['delete_bin'] = False
    ops['look_one_level_down'] = True
    ops['mesoscan'] = False
    ops['save_path0'] = save_path
    ops['save_folder'] = []
    ops['move_bin'] = False # if 1, and fast_disk is different than save_disk, binary file is moved to save_disk
    ops['combined'] = True

    # recording params
    ops['nplanes'] = recording_params.get('nplanes',1)
    ops['nchannels'] = recording_params.get('nchannels',1)
    ops['tau'] = recording_params.get('tau',1.33)
    ops['fs'] = recording_params.get('fs','fs')
    ops['aspect'] = recording_params.get('aspect',1.0) 
    # um/pixels in X / um/pixels in Y (for correct aspect ratio in GUI ONLY)

    # bidirectional phase offset correction
    ops['do_bidiphase'] = False

    # registration
    ops['do_registration'] = 1 # 2 forces re-registration
    ops['two_step_registration'] = False
    ops['nonrigid'] = True
    ops['reg_tif'] = True

    # cell detection
    ops['roidetect'] = True
    ops['spikedetect'] = True
    ops['sparse_mode'] = True # not clear what this does? something about extracting sparsely active cells activities
    ops['connected'] = True #whether or not to require ROIs to be fully connected (set to 0 for dendrites/boutons)
    ops['threshold scaling'] = 5.0
    ops['max_overlap'] = 0.75
    ops['high_pass'] = 100 #running mean subtraction across time with window of size 'high_pass'. Values of less than 10 are 
                           #recommended for 1P data where there are often large full-field changes in brightness.
    ops['smooth_masks'] = True # whether to smooth masks in final pass of cell detection. This is useful especially if you are in a high noise regime.
    ops['max_iterations'] = 20
    ops['nbinned'] = 5000 #maximum number of binned frames to use for ROI detection.

    # signal extraction
    ops['min_neuropil_pixels'] = 350
    ops['inner_neuropil_radius'] = 2

    # spike deconvolution
    # We neuropil-correct the trace Fout = F - ops['neucoeff'] * Fneu, 
    # and then baseline-correct these traces with an ops['baseline'] filter, and then detect spikes.
    ops['neucoeff'] = 0.7

    # filtering the data with a Gaussian of width ops['sig_baseline'] * ops['fs'], 
    # then minimum filtering with a window of ops['win_baseline'] * ops['fs'], and then maximum filtering with the same window.
    ops['baseline'] = 'maximin'
    ops['win_baseline'] = 60.0 #window for maximin filter in seconds
    ops['sig_baseline'] = 10.0 # window for gaussian filter in seconds

    # # filtering with a Gaussian of width ops['sig_baseline'] * ops['fs'] and then taking the minimum
    # ops['baseline'] = 'constant'
    # ops['sig_baseline'] = 10.0 # window for gaussian filter in seconds

    # # constant baseline by taking the ops['prctile_baseline'] percentile of the trace
    # ops['baseline'] = 'constant_percentile'
    # ops['prctile_baseline'] = 8
    
    for k,v in other_params.items():
        ops[k] = v
        print("Setting %s: %s" % (str(k), str(v)))

    return ops

def create_shmem(shmem_params):
    shmem = shared_memory.SharedMemory(create=True,size=shmem_params['nbytes'])
    shmem_params['name'] = shmem.name
    return shmem, shmem_params

def create_shmem_from_arr(sample_arr, copy=False):
    shmem_params = {
        'dtype' : sample_arr.dtype,
        'shape' : sample_arr.shape,
        'nbytes' : sample_arr.nbytes
         }
    shmem, shmem_params = create_shmem(shmem_params)
    sh_arr = n.ndarray(shmem_params['shape'], shmem_params['dtype'],
                        buffer = shmem.buf)
    if copy:
        sh_arr[:] = sample_arr[:]
    else:
        sh_arr[:] = 0

    return shmem, shmem_params, sh_arr

def load_shmem(shmem_params):
    shmem = shared_memory.SharedMemory(name=shmem_params['name'], create=False)
    sh_arr = n.ndarray(shmem_params['shape'], shmem_params['dtype'],
                        buffer = shmem.buf)
    return shmem, sh_arr

def close_shmem(shmem_params):
    shmem = shared_memory.SharedMemory(name=shmem_params['name'], create=False)
    shmem.close()
def close_and_unlink_shmem(shmem_params):
    print("Don't use me. I cause memory leaks :(")
    if 'name' in shmem_params.keys():
        shmem = shared_memory.SharedMemory(name=shmem_params['name'], create=False)
        shmem.close()
        shmem.unlink()

def get_centroid(ref_img_3d):
    mean_im = ref_img_3d.mean(axis=0)
    M = moments(mean_im, order=1)
    centroid = (int(M[1, 0] / M[0, 0]), int(M[0, 1] / M[0, 0]))
    return centroid
def pad_crop_movie(mov, centroid, crop_size):
    mov = mov[:,:,centroid[0]-crop_size[0]//2:centroid[0]+crop_size[0]//2,
                    centroid[1]-crop_size[1]//2:centroid[1]+crop_size[1]//2]
    nyy, nxx = mov.shape[2:]
    pad = [(0,0), (0,0), (0,0), (0,0)]
    do_pad = False
    if nyy < crop_size[0]:
        pad[2] = (0, crop_size[0] - nyy)
        do_pad = True
    if nxx < crop_size[1]:
        pad[3] = (0, crop_size[1] - nxx)
        do_pad = True
    if do_pad:
        mov = n.pad(mov, pad)
    return mov
    

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
