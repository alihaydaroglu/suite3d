import time
from dask_image.ndfilters import uniform_filter as dask_uniform_filter
import cProfile
import multiprocessing
import numpy as n

np = n
from dask import array as darr
from scipy.ndimage import maximum_filter, gaussian_filter, uniform_filter

from . import utils
from .utils import default_log

# This file is adapted from the original suite2p, many functions are substantially the same


def accumulate_mean(mean, batch, ns_previous, sample_axis=0):
    """
    Add a new batch of samples to a running mean (inplace).

    Args:
        mean (ndarray): mean of each voxel across all previous batches
        batch (ndarray): ns_batch new samples of size mean.shape
        ns_previous (int): number of samples before the current batch
        sample_axis (int, optional): Axis along which samples are stacked
                                     in batch. Defaults to 0.

    Returns:
        ndarray: new mean (array has been modified inplace)
    """
    ns_batch = batch.shape[sample_axis]
    ns_total = ns_batch + ns_previous
    batch_mean = batch.mean(axis=sample_axis)
    mean *= (ns_previous) / ns_total
    mean += batch_mean * (ns_batch / ns_total)
    return mean


def accumulate_max(maxx, batch, sample_axis=0):
    """
    Add a new batch of samples to running maximum (inplace)

    Args:
        maxx (ndarray): maximum value for each voxel across previous batches
        batch (_type_): ns_batch new samples of size mean.shape
        sample_axis (int, optional):Axis along which samples are stacked
                                     in batch. Defaults to 0.

    Returns:
        maxx: new max for each voxel (array has been mofidied inplace)
    """
    batch_max = batch.max(axis=sample_axis)
    update_indices = batch_max > maxx
    maxx[update_indices] = batch_max[update_indices]
    return maxx


def accumulate_sdmov(sdmov_2, batch, ns_previous, minibatch_size=500):
    """
    Add a new batch of samples to the computation of sdmov. sdmov is
    of shape nz,ny,nx, and its value at each voxel can be roughly interpreted
    as the 'jaggedness' or 'peakiness' of the timecourse of the voxel.

    First, we compute the difference between successive frames for each voxel.
    Then, we compute the standard deviation of these differences.
    To accumulate across batches, we keep sdmov_2, which is the sum-of-squares of these
    differences. After adding each batch, the square root of sdmov_2 is the standard
    deviation for all frames we have seen.

    Args:
        sdmov_2 (ndarray): (nz, ny, nx) the current sum-of-squares of
                            differences for each voxel
        batch (_type_): (nt,nz,ny,nx) new frames to include
        ns_previous (int): number of frames in previous batches
        minibatch_size (int, optional): Number of frames to compute at once.

    Returns:
        _type_: _description_
    """
    ns_batch = batch.shape[0]
    ns_total = ns_batch + ns_previous
    minibatch_size = min(ns_batch, minibatch_size)
    for i in range(0, ns_batch, minibatch_size):
        # compute the difference between successive frames for each voxel
        dt_batch = n.diff(batch[i : i + minibatch_size], axis=0)
        # squared sum of these across time, and accumulate into sdmov_2
        sdmov_2 += (dt_batch**2).sum(axis=0)
    sdmov = n.sqrt(n.maximum(1e-10, (sdmov_2 / ns_total)))
    return sdmov


def normalize_movie_by_sdmov(mov, sdmov, sdnorm_exp=1.0):
    """
    Divide each voxel of the movie by sdmov ** sdnorm_exp

    Args:
        mov (ndarray): (nt,nz,ny,nx)
        sdmov (ndarray): (nz,ny,nx)
        sdnorm_exp (float, optional): Defaults to 1.0.

    Returns:
        mov: normalized movie (edited inplace)
    """
    sdnorm = sdmov**sdnorm_exp
    mov /= sdnorm
    return mov


def filter_and_reduce_movie(
    mov,
    npil_filt_type,
    npil_filt_size,
    cell_filt_type,
    cell_filt_size,
    intensity_thresh,
    n_proc=8,
    minibatch_size=20,
    standard_vmap=True,
    log=default_log,
):
    """
    Apply neuropil subtraction and cell deteciton filters to movie. Then, threshold
    each pixel, and take the squared sum of times where it exceeds the threshold.
    Optimized in shared memory.

    Args:
        mov (ndarray): nt,nz,ny,nx
        npil_filt_type (str): unif for uniform, gaussian for gaussian filter
        npil_filt_size (tuple): 3-tuple of sizes in z,y,x for the filters
        cell_filt_type (str): same as npil_filt_type
        cell_filt_size (tuple): same as cell_filt_size
        intensity_thresh (float): Threshold above which pixel is considered active on a frame
        n_proc (int, optional): Number of processors. Defaults to 8.
        minibatch_size (int, optional): Number of frames to give each processor. Defaults to 20.
        log (func, optional): Defaults to default_log.

    Returns:
        vmap_2, mov_sub
    """

    log("Loading movie into shared memory", 3)
    log("dtu_shmem", tic=True)
    # Load a copy of the movie into shared memory, and delete the original
    shmem_mov_sub, shmem_par_mov_sub, mov_sub = utils.create_shmem_from_arr(
        mov, copy=True
    )
    del mov
    # Create another array in shared memory for the neuropil-subtracted movie
    shmem_mov_filt, shmem_par_mov_filt, mov_filt = utils.create_shmem_from_arr(
        mov_sub, copy=False
    )
    log("dtu_shmem", toc=True)
    log("Subtracting neuropil and applying cell filters", 3)
    log("dtu_npsub_conv3d", tic=True)
    np_sub_and_conv3d_split_shmem(
        shmem_par_mov_sub,
        shmem_par_mov_filt,
        npil_filt_size,
        cell_filt_size,
        n_proc=n_proc,
        batch_size=minibatch_size,
        np_filt_type=npil_filt_type,
        conv_filt_type=cell_filt_type,
    )
    log("dtu_npsub_conv3d", toc=True)
    log("Reducing filtered movie to compute correlation map", 3)
    log("dtu_vmap", tic=True)

    if standard_vmap:
        vmap_2 = get_vmap3d(
            mov_filt, intensity_thresh, sqrt=False, mean_subtract=False, fix_edges=False
        )
    else:
        log("RUNNING NEW VMAP, 2")
        vmap_2 = get_vmap3d_cov(mov_filt, mov_sub, thresh=intensity_thresh)

    log("dtu_vmap", toc=True)
    # close and free the shared memory arrays
    log("dtu_cleanup", tic=True)
    shmem_mov_filt.close()
    shmem_mov_filt.unlink()
    # before closing the mov_sub shared memory, make a copy of it to return
    mov_sub_return = mov_sub.copy()
    shmem_mov_sub.close()
    shmem_mov_sub.unlink()
    log("dtu_cleanup", toc=True)
    return vmap_2, mov_sub_return


def accumulate_vmap_2(vmap_2, new_vmap_2, ns_total):
    vmap_2 += new_vmap_2
    vmap = n.sqrt(n.maximum(1e-10, vmap_2 / ns_total))
    return vmap


def get_vmap3d_cov(mov1, mov2, thresh=0):
    if thresh is None:
        thresh = 0
    nt, nz, ny, nx = mov1.shape
    vmap = n.zeros((nz, ny, nx))
    for tidx in range(nt):
        active_pix = mov1[tidx] > thresh
        # print(mov1.shape)
        # print(mov2.shape)
        # print(mov1[tidx][active_pix].shape)
        # print(vmap.shape)
        vmap[active_pix] += mov1[tidx][active_pix] * mov2[tidx][active_pix]
    return vmap


def threshold_reduce(
    mov: np.ndarray,
    intensity_threshold: float,
    fix_edges=False,
    mean_subtract=False,
    sqrt=True,
):
    """
    Returns standard deviation of pixels, thresholded by 'intensity_threshold'.
    Run in a loop to reduce memory footprint.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to downsample
    intensity_threshold: float
        The threshold to use

    Returns
    -------
    Vt: Ly x Lx
        The standard deviation of the non-thresholded pixels
    """
    nbinned, Lyp, Lxp = mov.shape
    Vt = np.zeros((Lyp, Lxp), "float32")
    if mean_subtract:
        mov = mov.copy() - mov.mean(axis=0)
    # print(f"Intensity thresh is {intensity_threshold}")
    if intensity_threshold is None:
        intensity_threshold = mov.min()

    # print(f"Intensity thresh is {intensity_threshold}")
    # print(f"Mov mean is {mov.mean()}")
    for t in range(nbinned):
        Vt += mov[t] ** 2 * (mov[t] > intensity_threshold)
    if sqrt:
        Vt = Vt**0.5
    if fix_edges:
        # print("FIXING EDGES")
        Vt[0] = Vt[1]
        Vt[-1] = Vt[-2]
        Vt[:, 0] = Vt[:, 1]
        Vt[:, -1] = Vt[:, -2]
    # else:
    # print("NOT FIXING EDGES")
    return Vt


def binned_mean(mov: np.ndarray, bin_size) -> np.ndarray:
    """Returns an array with the mean of each time bin (of size 'bin_size')."""
    n_frames, Lz, Ly, Lx = mov.shape
    mov = mov[: (n_frames // bin_size) * bin_size]
    return mov.reshape(-1, bin_size, Lz, Ly, Lx).astype(np.float32).mean(axis=1)


def standard_deviation_over_time(
    mov: np.ndarray, batch_size: int, sqrt=True, dask=False
) -> np.ndarray:
    """
    Returns standard deviation of difference between pixels across time, computed in batches of batch_size.

    Parameters
    ----------
    mov: nImg x Lz x Ly x Lx
        The frames to filter
    batch_size: int
        The batch size

    Returns
    -------
    filtered_mov: Lz x Ly x Lx
        The statistics for each pixel
    """
    nbins = mov.shape[0]
    batch_size = min(batch_size, nbins)
    sdmov = np.zeros(mov.shape[1:], "float32")
    if dask:
        sdmov = darr.zeros(*mov.shape[1:], dtype=n.float32)
    for ix in range(0, nbins, batch_size):
        sdmov += (np.diff(mov[ix : ix + batch_size], axis=0) ** 2).sum(axis=0)
    if sqrt:
        sdmov = np.sqrt(np.maximum(1e-10, (sdmov / nbins)))
    return sdmov


# def npsub_worker(filt_size, mode, c1):
#     frame = n.zeros((8,1202,1015))
#     return frame - uniform_filter(frame, size=filt_size, mode=mode) / c1
# return frame - frame / c1
def npsub_worker(mov_params, idxs, filt_size, mode, c1):
    print("RUNNING idxs: %s" % str(idxs))
    cp = cProfile.Profile()
    cp.enable()
    shmem, mov = utils.load_shmem(mov_params)
    for idx in idxs:
        # mov[idx] = mov[idx] - (uniform_filter(mov[idx], size=filt_size, mode = mode) / c1)
        mov[idx] = mov[idx] - mov[idx] / c1
    cp.disable()
    cp.print_stats(sort="cumtime")


def neuropil_subtraction(
    mov: np.ndarray, filter_size: int, filter_size_z: int, mode="constant"
) -> None:
    """Returns movie subtracted by a low-pass filtered version of itself to help ignore neuropil."""

    nt, Lz, Ly, Lx = mov.shape
    filt_size = (filter_size_z, filter_size, filter_size)
    # print('Neuropil filter size:', filt_size)
    c1 = uniform_filter(np.ones((Lz, Ly, Lx)), size=filt_size, mode=mode)
    movt = np.zeros_like(mov)
    for frame, framet in zip(mov, movt):
        framet[:] = frame - (uniform_filter(frame, size=filt_size, mode=mode) / c1)
    return movt


def neuropil_subtraction_dask(
    mov: np.ndarray, filter_size: int, filter_size_z: int, mode="constant"
) -> None:
    """Returns movie subtracted by a low-pass filtered version of itself to help ignore neuropil."""

    nt, Lz, Ly, Lx = mov.shape
    filt_size = (filter_size_z, filter_size, filter_size)
    # print('Neuropil filter size:', filt_size)
    c1 = uniform_filter(np.ones((Lz, Ly, Lx)), size=filt_size, mode=mode)

    for i in range(nt):
        mov[i] = mov[i] - dask_uniform_filter(mov[i], size=filt_size, mode=mode) / c1
    return mov


def neuropil_subtraction_debug(
    mov: np.ndarray, filter_size: int, filter_size_z: int, mode="constant"
) -> None:
    """Returns movie subtracted by a low-pass filtered version of itself to help ignore neuropil."""

    nt, Lz, Ly, Lx = mov.shape
    filt_size = (filter_size_z, filter_size, filter_size)
    # print('Neuropil filter size:', filt_size)
    c1 = uniform_filter(np.ones((Lz, Ly, Lx)), size=filt_size, mode=mode)
    movt = np.zeros_like(mov)
    lpfs = []
    for frame, framet in zip(mov, movt):
        lpf = uniform_filter(frame, size=filt_size, mode=mode) / c1
        framet[:] = frame - lpf
        lpfs.append(lpf)
    return movt, n.array(lpfs)


def downsample(mov: np.ndarray, taper_edge: bool = True) -> np.ndarray:
    """
    Returns a pixel-downsampled movie from 'mov', tapering the edges of 'taper_edge' is True.

    Parameters
    ----------
    mov: nImg x Lz x Ly x Lx
        The frames to downsample
    taper_edge: bool
        Whether to taper the edges

    Returns
    -------
    filtered_mov:
        The downsampled frames
    """
    n_frames, Lz, Ly, Lx = mov.shape

    # bin along Y
    movd = np.zeros((n_frames, Lz, int(np.ceil(Ly / 2)), Lx), "float32")
    movd[:, :, : Ly // 2, :] = np.mean([mov[:, :, 0:-1:2, :], mov[:, :, 1::2, :]], axis=0)
    if Ly % 2 == 1:
        movd[:, :, -1, :] = mov[:, :, -1, :] / 2 if taper_edge else mov[:, :, -1, :]

    # bin along X
    mov2 = np.zeros((n_frames, Lz, int(np.ceil(Ly / 2)), int(np.ceil(Lx / 2))), "float32")
    mov2[:, :, :, : Lx // 2] = np.mean(
        [movd[:, :, :, 0:-1:2], movd[:, :, :, 1::2]], axis=0
    )
    if Lx % 2 == 1:
        mov2[:, :, :, -1] = movd[:, :, :, -1] / 2 if taper_edge else movd[:, :, :, -1]

    return mov2


def square_convolution_2d(
    mov: np.ndarray, filter_size: int, filter_size_z: int
) -> np.ndarray:
    """Returns movie convolved by uniform kernel with width 'filter_size'."""
    movt = np.zeros_like(mov, dtype=np.float32)
    filt_size = (filter_size_z, filter_size, filter_size)
    for frame, framet in zip(mov, movt):
        framet[:] = filter_size * uniform_filter(frame, size=filt_size, mode="constant")
    return movt


def get_vmap3d(
    movu0, intensity_threshold=None, fix_edges=False, sqrt=True, mean_subtract=True
):
    nt, nz, ny, nx = movu0.shape
    vmap = n.zeros((nz, ny, nx))
    for i in range(nz):
        vmap[i] = threshold_reduce(
            movu0[:, i],
            intensity_threshold,
            mean_subtract=mean_subtract,
            fix_edges=fix_edges,
            sqrt=sqrt,
        )
    return vmap


def get_vmap3d_shmem_w(shmem_in, shmem_vmap, z_idx, intensity_threshold, fix_edges, sqrt):
    shin, mov_in = utils.load_shmem(shmem_in)
    shvmap, vmap_z = utils.load_shmem(shmem_vmap)

    vmap_z[z_idx] = threshold_reduce(
        mov_in[:, z_idx], intensity_threshold, fix_edges, sqrt
    )


def get_vmap3d_shmem(
    shmem_in,
    shmem_vmap,
    intensity_threshold=None,
    fix_edges=True,
    sqrt=True,
    n_proc=15,
    pool=None,
):
    nt, nz, ny, nx = shmem_in["shape"]
    if pool is None:
        pool = multiprocessing.Pool(n_proc)
    pool.starmap(
        get_vmap3d_shmem_w,
        [
            (shmem_in, shmem_vmap, z_idx, intensity_threshold, fix_edges, sqrt)
            for z_idx in range(nz)
        ],
    )


def np_sub_and_conv3d_shmem_w(
    in_par, idxs, np_filt_size, conv_filt_size, c1, np_filt, conv_filt
):
    shin, mov_in = utils.load_shmem(in_par)
    for idx in idxs:
        mov_in[idx] = mov_in[idx] - (
            np_filt(mov_in[idx], size=np_filt_size, mode="constant") / c1
        )
        mov_in[idx] = conv_filt(mov_in[idx], size=conv_filt_size, mode="constant")


def np_sub_and_conv3d_shmem(
    shmem_in,
    np_filt_size,
    conv_filt_size,
    n_proc=8,
    batch_size=50,
    pool=None,
    np_filt_type="unif",
    conv_filt_type="unif",
):
    """
    WARNING: this is not optimal because the process startup time is quite long for each process
    seems like this is because each subprocess imports suite2p, which takes about 1-2 seconds
    this should all be in parallel, but in reality it causes memory bottlenecks so lasts about ~10 seconsds
    so if you have a mp.pool().starmap() that does nothing, it takes 10 seconds to run
    this is because windows does not have forking and python uses the "spawn" start method
    https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    in the future: either maintain the same processes for the whole thing, or figure out something else
    """
    nt, Lz, Ly, Lx = shmem_in["shape"]
    if np_filt_type == "unif":
        np_filt = uniform_filter
    elif np_filt_type == "gaussian":
        np_filt = gaussian_filter

    if conv_filt_type == "unif":
        conv_filt = uniform_filter
    elif conv_filt_type == "gaussian":
        conv_filt = gaussian_filter

    c1 = uniform_filter(n.ones((Lz, Ly, Lx)), np_filt_size, mode="constant")

    batches = [
        n.arange(idx, min(nt, idx + batch_size)) for idx in n.arange(0, nt, batch_size)
    ]
    close = True
    if pool is None:
        pool = multiprocessing.Pool(n_proc)
        close = False
    pool.starmap(
        np_sub_and_conv3d_shmem_w,
        [
            (shmem_in, b, np_filt_size, conv_filt_size, c1, np_filt, conv_filt)
            for b in batches
        ],
    )
    if close:
        pool.close()
        pool.terminate()


filter_mode = "mirror"


def np_sub_and_conv3d_split_shmem_w(
    sub_par, filt_par, idxs, np_filt_size, conv_filt_size, c1, c2, np_filt, conv_filt
):
    # print(idxs)
    tic = time.time()
    sub_sh, mov_sub = utils.load_shmem(sub_par)
    filt_sh, mov_filt = utils.load_shmem(filt_par)
    # print(time.time() - tic)
    # tic = time.time()
    for idx in idxs:
        if np_filt is not None:
            mov_sub[idx] = mov_sub[idx] - (
                np_filt(
                    mov_sub[idx], np_filt_size, mode=filter_mode, cval=mov_sub[idx].mean()
                )
                / c1
            )
        mov_filt[idx] = conv_filt_size[-1] * conv_filt(
            mov_sub[idx], conv_filt_size, mode=filter_mode, cval=mov_sub[idx].mean()
        )  # / c2
    sub_sh.close()
    filt_sh.close()
    # print("WORKER COMPLETE IN ", time.time() - tic)


def np_sub_and_conv3d_split_shmem(
    shmem_sub,
    shmem_filt,
    np_filt_size,
    conv_filt_size,
    n_proc=8,
    batch_size=50,
    pool=None,
    np_filt_type="unif",
    conv_filt_type="unif",
):
    tic = time.time()
    nt, Lz, Ly, Lx = shmem_sub["shape"]
    if np_filt_type == "unif":
        np_filt = uniform_filter
    elif np_filt_type == "gaussian":
        np_filt = gaussian_filter
    elif np_filt_type == "none" or np_filt_type is None:
        c1 = 1
        np_filt = None
    else:
        assert False  # np_filt = None

    if conv_filt_type == "unif":
        conv_filt = uniform_filter
    elif conv_filt_type == "gaussian":
        conv_filt = gaussian_filter
    else:
        conv_filt = None
    if np_filt is not None:
        c1 = np_filt(n.ones((Lz, Ly, Lx)), np_filt_size, mode=filter_mode)
    c2 = conv_filt(n.ones((Lz, Ly, Lx)), conv_filt_size, mode=filter_mode)

    batches = [
        n.arange(idx, min(nt, idx + batch_size)) for idx in n.arange(0, nt, batch_size)
    ]
    close = True
    if pool is None:
        pool = multiprocessing.Pool(n_proc)
        close = False
    # print(f"setup in {time.time() - tic}s")
    # print(batches)
    # print("%d batches, %d batchsize, %d proc" % (len(batches), batch_size, n_proc))
    pool.starmap(
        np_sub_and_conv3d_split_shmem_w,
        [
            (
                shmem_sub,
                shmem_filt,
                b.astype(int),
                np_filt_size,
                conv_filt_size,
                c1,
                c2,
                np_filt,
                conv_filt,
            )
            for b in batches
        ],
    )
    if close:
        pool.close()
        pool.terminate()


def np_sub_shmem_w(in_par, idxs, np_filt_size, c1):
    shin, mov_in = utils.load_shmem(in_par)
    for idx in idxs:
        mov_in[idx] = mov_in[idx] - (
            uniform_filter(mov_in[idx], size=np_filt_size, mode="constant") / c1
        )


def np_sub_shmem(shmem_in, np_filt_size, n_proc=8, batch_size=50, pool=None):
    """
    WARNING: this is not optimal because the process startup time is quite long for each process
    seems like this is because each subprocess imports suite2p, which takes about 1-2 seconds
    this should all be in parallel, but in reality it causes memory bottlenecks so lasts about ~10 seconsds
    so if you have a mp.pool().starmap() that does nothing, it takes 10 seconds to run
    this is because windows does not have forking and python uses the "spawn" start method
    https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    in the future: either maintain the same processes for the whole thing, or figure out something else
    """
    nt, Lz, Ly, Lx = shmem_in["shape"]
    c1 = uniform_filter(n.ones((Lz, Ly, Lx)), np_filt_size, mode="constant")

    batches = [
        n.arange(idx, min(nt, idx + batch_size)) for idx in n.arange(0, nt, batch_size)
    ]
    close = True
    if pool is None:
        pool = multiprocessing.Pool(n_proc)
        close = False
    pool.starmap(np_sub_shmem_w, [(shmem_in, b, np_filt_size, c1) for b in batches])
    if close:
        pool.close()
        pool.terminate()


def hp_rolling_mean_filter(mov: np.ndarray, width: int, copy=True) -> np.ndarray:
    """
    Returns a high-pass-filtered copy of the 3D array 'mov' using a non-overlapping rolling mean kernel over time.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to filter
    width: int
        The filter width

    Returns
    -------
    filtered_mov: nImg x Ly x Lx
        The filtered frames

    """
    if copy:
        mov = mov.copy()
    for i in range(0, mov.shape[0], width):
        mov[i : i + width,] -= mov[i : i + width].mean(axis=0)
    return mov


def hp_rolling_mean_filter_mp(shmem_par, width, nz, n_proc=16):
    pool = multiprocessing.Pool(n_proc)
    pool.starmap(
        hp_rolling_mean_filter_shmem_w, [(shmem_par, z_idx, width) for z_idx in range(nz)]
    )
    pool.close()
    pool.terminate()


def hp_rolling_mean_filter_shmem_w(shmem_par, z_idx, width) -> np.ndarray:
    if z_idx == 0:
        tic = time.time()
    mov_sh, mov = utils.load_shmem(shmem_par)
    if z_idx == 0:
        print(time.time() - tic)
    for i in range(0, mov.shape[0], width):
        mov[i : i + width, z_idx] -= mov[i : i + width, z_idx].mean(axis=0)

    if z_idx == 0:
        print(time.time() - tic)
    mov_sh.close()


# def np_sub_shmem_w(in_par, out_par, idxs, size, c1):
#     shin, mov_in = utils.load_shmem(in_par)
#     shout, mov_out = utils.load_shmem(out_par)
#     for idx in idxs:
#         mov_out[idx] = mov_in[idx] - \
#             (uniform_filter(mov_in[idx], size=size, mode='constant') / c1)


# def np_sub_shmem(shmem_in, shmem_out, size, n_proc, batch_size=50):
#     nt, Lz, Ly, Lx = shmem_in['shape']
#     c1 = uniform_filter(n.ones((Lz, Ly, Lx)), size, mode='constant')

#     batches = [n.arange(idx, min(nt, idx+batch_size))
#                for idx in n.arange(0, nt, batch_size)]
#     pool = multiprocessing.Pool(n_proc)
#     pool.starmap(np_sub_shmem_w, [
#                  (shmem_in, shmem_out, b, size, c1) for b in batches])
