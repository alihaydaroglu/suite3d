import os
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import (
    maximum_filter,
    gaussian_filter,
    uniform_filter,
    percentile_filter,
)
import numpy as n
from multiprocessing import Pool
from . import utils
from scipy.spatial import distance_matrix
from .utils import default_log

from skimage.filters import threshold_local


def detect_cells(
    patch,
    vmap,
    max_iter=10000,
    peak_thresh=2.5,
    activity_thresh=2.5,
    extend_thresh=0.2,
    allow_overlap=True,
    roi_ext_iterations=2,
    max_ext_iters=20,
    percentile=0,
    log=default_log,
    recompute_v=False,
    offset=(0, 0, 0),
    savepath=None,
    debug=False,
    patch_idx=-1,
    **kwargs,
):
    nt, nz, ny, nx = patch.shape
    stats = []

    Th2 = activity_thresh
    vmultiplier = 1  # max(1, nt / magic_number)
    peak_thresh = vmultiplier * peak_thresh
    vmin = vmap.min()
    log(
        "Starting extraction with peak_thresh: %0.3f and Th2: %0.3f" % (peak_thresh, Th2),
        2,
    )

    for iter_idx in range(max_iter):
        med, zz, yy, xx, lam, peak_val = find_top_roi3d(vmap, xy_pix_scale=3)
        if peak_val < peak_thresh:
            log(
                "Iter %04d: peak is too small (%0.3f) - ending extraction"
                % (iter_idx, peak_val),
                2,
            )
            break
        tproj = patch[:, zz, yy, xx] @ lam
        threshold = min(Th2, n.percentile(tproj, percentile)) if percentile > 0 else Th2
        active_frames = n.nonzero(tproj > threshold)[0]

        for i in range(roi_ext_iterations):
            log("%d/%d active frames" % (len(active_frames), nt), 3)
            if len(active_frames) == 0:
                log("WARNING: no active frames in roi %d" % iter_idx, 1)
                log("WILL BREAK")
                break

            zz, yy, xx, lam = iter_extend3d(
                zz,
                yy,
                xx,
                active_frames,
                patch,
                extend_thresh=extend_thresh,
                max_ext_iters=max_ext_iters,
                verbose=debug,
            )
            tproj = patch[:, zz, yy, xx] @ lam
            # print("           active frames before recompute: %d" % len(active_frames))
            active_frames = n.nonzero(tproj > threshold)[0]
            # print("           active frames after recompute: %d" % len(active_frames))
            npix = len(lam)
        if len(active_frames) == 0:
            log("BREAKING")
            continue
        sub = n.zeros((nt, npix))
        sub[active_frames] = tproj[active_frames, n.newaxis] @ lam[n.newaxis]
        patch[:, zz, yy, xx] -= sub

        if allow_overlap:
            print("Warning: this is not good")
            # should properly recompute vmap using the convovled movie, not just the subtracted movie
            # see lines with multiscale_mask where movu is edited in sparsery
            # TODO
            mnew = patch[:, zz, yy, xx]
            vmap[zz, yy, xx] = ((mnew**2) * n.float32(mnew > threshold)).sum(
                axis=0
            ) ** 0.5
        else:
            zzx, yyx, xxx = extend_roi3d(zz, yy, xx, (nz, ny, nx), extend_z=True)
            zzx, yyx, xxx = extend_roi3d_iter(
                zzx, yyx, xxx, (nz, ny, nx), n_iters=0, extend_z=False
            )
            # print(zz)
            # print(zzx)
            vmap[zzx, yyx, xxx] = vmin

        stat = {
            "idx": iter_idx,
            "coords_patch": (zz, yy, xx),
            "coords": (zz + offset[0], yy + offset[1], xx + offset[2]),
            "lam": lam,
            "med_patch": med,
            "med": (med[0] + offset[0], med[1] + offset[1], med[2] + offset[2]),
            "active_frames": active_frames,
            "patch_idx": patch_idx,
        }
        stats.append(stat)
        #
        # log("Cell %d activity_thresh %.3f, peak_thresh: %.3f, %d active_frames" % (iter_idx+1, threshold, peak_thresh, len(active_frames)), 2)
        log(
            "Added cell %d at %02d, %03d, %03d, peak: %0.3f, %d frames, %d pixels"
            % (
                len(stats),
                stat["med"][0],
                stat["med"][1],
                stat["med"][2],
                peak_val,
                len(active_frames),
                npix,
            ),
            3,
        )
        if savepath is not None and iter_idx % 250 == 0 and iter_idx > 0:
            n.save(savepath, stats)
            log("Saving checkpoint to %s" % savepath)
    log("Found %d cells in %d iterations" % (len(stats), iter_idx + 1), 1)
    if savepath is not None:
        log("Saving cells to %s" % savepath, 1)
        n.save(savepath, stats)
        is_cell_path = savepath[:-9] + "iscell.npy"
        from .utils import make_iscell, save_iscell
        log("Saving iscell.npy to %s" % is_cell_path, 1)
        save_iscell(is_cell_path, make_iscell(len(stats)))
    return stats


def detect_cells_mp(
    patch,
    vmap,
    n_proc_detect=8,
    max_iter=10000,
    peak_thresh=2.5,
    activity_thresh=2.5,
    extend_thresh=0.2,
    roi_ext_iterations=2,
    max_ext_iters=20,
    percentile=0,
    ext_subtract_iters=2,
    log=default_log,
    max_pix=250,
    recompute_v=False,
    allow_overlap=False,
    offset=(0, 0, 0),
    savepath=None,
    extension_func = 'corr',
    debug=False,
    patch_idx=-1,
    **kwargs,
):
    """
    Detect cells in a 3D patch using multiprocessing.

    Args:
        patch (np.ndarray): 4D array of image data (time, z, y, x)
        vmap (np.ndarray): 3D array of variance map
        n_proc_detect (int): Number of processes to use for detection
        max_iter (int): Maximum number of iterations
        peak_thresh (float): Threshold for peak detection
        activity_thresh (float): Threshold for activity detection
        extend_thresh (float): Threshold for ROI extension
        roi_ext_iterations (int): Number of iterations for ROI extension
        max_ext_iters (int): Maximum number of extension iterations
        percentile (float): Percentile for thresholding
        log (function): Logging function
        max_pix (int): Maximum number of pixels in an ROI
        recompute_v (bool): Whether to recompute variance map
        allow_overlap (bool): Whether to allow overlapping ROIs
        offset (tuple): Offset for coordinates
        savepath (str): Path to save results
        debug (bool): Whether to run in debug mode
        patch_idx (int): Index of the current patch

    Returns:
        list: List of detected cell statistics
    """
    stats = []
    log("Loading movie patch to shared memory", 3)
    shmem_patch, shmem_par_patch, patch = utils.create_shmem_from_arr(patch, copy=True)
    patch_norms = n.sqrt((patch**2).sum(axis=0))
    log("Loaded", 3)
    Th2 = activity_thresh
    vmultiplier = 1
    peak_thresh = vmultiplier * peak_thresh
    vmin = vmap.min()
    log(f"Starting extraction with peak_thresh: {peak_thresh:.3f} and Th2: {Th2:.3f}", 2)
    nt, nz, ny, nx = patch.shape
    n_iters = max_iter // n_proc_detect
    roi_idx = 0
    widxs = n.arange(n_proc_detect)
    
    with Pool(n_proc_detect) as p:
        prev_n_rois = 0
        for iter_idx in range(n_iters):
            n_rois = len(stats)
            outs = find_top_n_rois(vmap, n_rois=n_proc_detect)
            filtered_rois = filter_rois(outs, peak_thresh)

            if not filtered_rois:
                log(f"Iter {iter_idx:04d}: peak is too small - ending extraction", 2)
                break

            log(
                f"Iter {iter_idx:04d}: running {len(filtered_rois):02d} ROIs in parallel",
                3,
            )
            roi_idxs = n.arange(len(filtered_rois)) + roi_idx + 1

            returns = p.starmap(
                detect_cells_worker,
                [
                    (
                        widxs[i],
                        roi_idxs[i],
                        shmem_par_patch,
                        filtered_rois[i],
                        Th2,
                        percentile,
                        roi_ext_iterations,
                        extend_thresh,
                        max_ext_iters,
                        offset,
                        max_pix,
                        patch_idx,
                        patch_norms,
                        extension_func,
                    )
                    for i in range(len(filtered_rois))
                ],
            )

            process_returns(
                returns,
                patch,
                vmap,
                stats,
                allow_overlap,
                vmin,
                savepath,
                log,
                ext_subtract_iters,
            )
            roi_idx = len(stats)
            n_rois = len(stats)
            n_rois_iter = n_rois - prev_n_rois
            prev_n_rois = n_rois
            log(f"Iter {iter_idx:04d}: added {n_rois_iter} ROIs, total {n_rois}", 2)
            if n_rois_iter == 0:
                log("No ROIs added this iteration - ending extraction", 2)
                break

            if savepath is not None and roi_idx % 250 == 0 and roi_idx > 0:
                save_checkpoint(savepath, stats, log)

    shmem_patch.close()
    shmem_patch.unlink()
    log(f"Found {roi_idx} cells in {iter_idx+1} iterations")
    save_final_results(savepath, stats, log)
    return stats


def filter_rois(outs, peak_thresh):
    """
    Filter ROIs based on peak threshold.

    Args:
        outs (list): List of ROI outputs
        peak_thresh (float): Threshold for peak values

    Returns:
        list: Filtered list of ROIs
    """
    return [out for out in outs if out[-1] >= peak_thresh]


def process_returns(
    returns, patch, vmap, stats, allow_overlap, vmin, savepath, log, ext_subtract_iters
):
    """
    Process the returns from cell detection workers.

    Args:
        returns (list): List of returns from workers
        patch (np.ndarray): 4D movie
        vmap (np.ndarray): 3D correlation map
        stats (list): List to store cell statistics
        allow_overlap (bool): Whether to allow overlapping ROIs
        vmin (float): Minimum value for vmap
        savepath (str): Path to save results
        log (function): Logging function
    """
    print("Processing returns")
    for batch_stats, batch_sub in returns:
        if batch_stats is None and batch_sub is None:
            continue
        zz, yy, xx = batch_stats["coords_patch"]
        threshold = batch_stats["threshold"]
        if len(zz) == 0:
            # this is kinda weird... why do some cells end up here? 
            continue
        patch[:, zz, yy, xx] -= batch_sub
        # print(len(zz), len(yy), len(xx))
        update_vmap(
            vmap, patch, zz, yy, xx, threshold, allow_overlap, vmin, ext_subtract_iters
        )
        stats.append(batch_stats)
        log_cell_addition(log, batch_stats, len(stats))


def update_vmap(
    vmap, patch, zz, yy, xx, threshold, allow_overlap, vmin, ext_subtract_iters=0
):
    """
    Update the corelatiton map after detecting a cell.

    Args:
        vmap (np.ndarray): 3D array of correlation map
        patch (np.ndarray): 4D array of image data
        zz, yy, xx (np.ndarray): Coordinates of the detected cell
        threshold (float): Activity threshold
        allow_overlap (bool): Whether to allow overlapping ROIs
        vmin (float): Minimum value for vmap
    """
    nz, ny, nx = vmap.shape
    if allow_overlap:
        mnew = patch[:, zz, yy, xx]
        vmap[zz, yy, xx] = (mnew * n.float32(mnew > threshold)).sum(axis=0) ** 0.5
    else:
        zzx, yyx, xxx = extend_roi3d(zz, yy, xx, vmap.shape, extend_z=True)

        zzx, yyx, xxx = extend_roi3d_iter(
            zzx, yyx, xxx, (nz, ny, nx), n_iters=ext_subtract_iters, extend_z=False
        )
        vmap[zzx, yyx, xxx] = vmin


def log_cell_addition(log, batch_stats, stats_len):
    """
    Log the addition of a new cell.

    Args:
        log (function): Logging function

        batch_stats (dict): Statistics of the detected cell
        stats_len (int): Current number of detected cells
    """
    med = batch_stats["med"]
    peak_val = batch_stats["peak_val"]
    npix = len(batch_stats["lam"])
    log(
        f"Added cell {stats_len} at {med[0]:02d}, {med[1]:03d}, {med[2]:03d}, peak: {peak_val:.3f}, {len(batch_stats['active_frames'])} frames, {npix} pixels",
        3,
    )


def save_checkpoint(savepath, stats, log):
    """
    Save a checkpoint of the current cell detection progress.

    Args:
        savepath (str): Path to save the checkpoint
        stats (list): List of cell statistics
        log (function): Logging function
    """
    n.save(savepath, stats)
    log(f"Saving checkpoint to {savepath}", 2)


def save_final_results(savepath, stats, log):
    """
    Save the final results of cell detection.

    Args:
        savepath (str): Path to save the results
        stats (list): List of cell statistics
        log (function): Logging function
    """
    if savepath is not None:
        log(f"Saving cells to {savepath}", 1)
        n.save(savepath, stats)
        is_cell_path = savepath[:-9] + "iscell.npy"
        from .utils import make_iscell, save_iscell
        log(f"Saving iscell.npy to {is_cell_path}", 1)
        save_iscell(is_cell_path, make_iscell(len(stats)))


def detect_cells_worker(
    worker_idx,
    roi_idx,
    patch_par,
    out,
    Th2,
    percentile,
    roi_ext_iterations,
    extend_thresh,
    max_ext_iters,
    offset,
    max_pix=1000,
    patch_idx=-1,
    patch_norms=None,
    extension_func = 'corr', # corr or proj
    ):
    """
    Worker function for detecting cells in parallel.

    Args:
        worker_idx: Index of the worker
        roi_idx: Index of the ROI being processed
        patch_par: Shared memory parameters for the patch
        out: Output from find_top_roi3d
        Th2: Activity threshold
        percentile: Percentile for thresholding
        roi_ext_iterations: Number of iterations for ROI extension
        extend_thresh: Threshold for ROI extension
        max_ext_iters: Maximum number of extension iterations
        offset: Offset for coordinates
        max_pix: Maximum number of pixels in an ROI
        patch_idx: Index of the current patch

    Returns:
        A tuple containing cell statistics and subtracted values
    """
    patch_sh, patch = utils.load_shmem(patch_par)
    med, zz, yy, xx, lam, peak_val = out
    tproj = patch[:, zz, yy, xx] @ lam
    threshold = min(Th2, n.percentile(tproj, percentile)) if percentile > 0 else Th2

    if extension_func == 'corr':
        extfunc = alternate_iter_extend3d
    else:
        extfunc = iter_extend3d

    active_frames = n.nonzero(tproj > threshold)[0]

    for i in range(roi_ext_iterations):
        if len(active_frames) == 0:
            default_log(
                "WARNING: no active frames in roi %d (iter %d) - if you keep getting this, increase peak_thresh, reduce percentile, or use a longer movie"
                % (roi_idx, i),
                1,
            )

            # print(n.percentile(tproj, 99.5))
            # print(n.percentile(tproj, 90))
            # print(n.percentile(tproj, 50))
            # default_log(f"Thresh: {threshold}, Max: {tproj.max()}")
            # assert False
            # default_log("RETURNING")
            return None, None
        zz, yy, xx, lam = extfunc(
            zz,
            yy,
            xx,
            active_frames,
            patch,
            extend_thresh=extend_thresh,
            max_ext_iters=max_ext_iters,
            max_pix=max_pix,
            patch_norms=patch_norms,
        )
        tproj = patch[:, zz, yy, xx] @ lam
        active_frames = n.nonzero(tproj > threshold)[0]
        npix = len(lam)

    sub = n.zeros((patch.shape[0], npix))
    sub[active_frames] = tproj[active_frames, n.newaxis] @ lam[n.newaxis]
    patch_sh.close()
    stat = {
        "idx": roi_idx,
        "threshold": threshold,
        "coords_patch": (zz, yy, xx),
        "coords": (zz + offset[0], yy + offset[1], xx + offset[2]),
        "lam": lam,
        "peak_val": peak_val,
        "med_patch": med,
        "med": (med[0] + offset[0], med[1] + offset[1], med[2] + offset[2]),
        "patch_idx": patch_idx,
        "active_frames": active_frames,
    }

    return stat, sub


def binned_mean(mov: n.ndarray, bin_size):
    """Returns an array with the mean of each time bin (of size 'bin_size')."""
    # adaptdet from suite2p/binary
    n_frames, nz, ny, nx = mov.shape
    mov = mov[: (n_frames // bin_size) * bin_size]
    return mov.reshape(-1, bin_size, nz, ny, nx).mean(axis=1)


def binned_mean_ax1(mov: n.ndarray, bin_size):
    """Returns an array with the mean of each bin (of size 'bin_size') along axis 1."""
    nz, n_frames, ny, nx = mov.shape
    mov = mov[:, : (n_frames // bin_size) * bin_size]
    return mov.reshape(nz, -1, bin_size, ny, nx).mean(axis=2)


def find_top_roi3d(V1, xy_pix_scale=3, z_pix_scale=1, peak_thresh=None):
    zi, yi, xi = n.unravel_index(n.argmax(V1), V1.shape)
    peak_val = V1.max()

    if peak_thresh is not None and peak_val < peak_thresh:
        print("Peak too small")
        return None, None, None, None, None, None

    zz, yy, xx, lam = add_square3d(zi, yi, xi, V1.shape, xy_pix_scale, z_pix_scale)

    med = (zi, yi, xi)
    return med, zz, yy, xx, lam, peak_val


def find_top_n_rois(
    V1, n_rois=5, xy_pix_scale=3, z_pix_scale=1, peak_thresh=None, vmin=0
):
    saves = []
    bufs = []
    outs = []
    V1 = V1.copy()  # does this break?
    for i in range(n_rois):
        med, zz, yy, xx, lam, peak_val = find_top_roi3d(
            V1, xy_pix_scale, z_pix_scale, peak_thresh
        )
        if med is None:
            bufs.append(None)
            saves.append(None)
        buf_zz, buf_yy, buf_xx, buf_lam = add_square3d(
            *med, V1.shape, xy_pix_scale=30, z_pix_scale=5
        )  # increased scale
        save = V1[buf_zz, buf_yy, buf_xx]
        saves.append(save)
        outs.append((med, zz, yy, xx, lam, peak_val))
        bufs.append((buf_zz, buf_yy, buf_xx))
        V1[buf_zz, buf_yy, buf_xx] = vmin
    for i in range(n_rois):
        if bufs[i] is not None:
            buf_zz, buf_yy, buf_xx = bufs[i]
            V1[buf_zz, buf_yy, buf_xx] = saves[i]
    return outs


def add_square3d(zi, yi, xi, shape, xy_pix_scale=3, z_pix_scale=1):
    nz, ny, nx = shape

    xs = n.arange(xi - int(xy_pix_scale / 2), xi + int(n.ceil(xy_pix_scale / 2)))
    ys = n.arange(yi - int(xy_pix_scale / 2), yi + int(n.ceil(xy_pix_scale / 2)))
    zs = n.arange(zi - int(z_pix_scale / 2), zi + int(n.ceil(z_pix_scale / 2)))
    zz, yy, xx = [vv.flatten() for vv in n.meshgrid(zs, ys, xs)]

    # check if each coord is within the possible coordinates
    valid_pix = n.all(
        [n.all([vv > -1, vv < nv], axis=0) for vv, nv in zip((zz, yy, xx), (nz, ny, nx))],
        axis=0,
    )

    zz = zz[valid_pix]
    yy = yy[valid_pix]
    xx = xx[valid_pix]

    mask = n.ones_like(zz)
    mask = mask / n.linalg.norm(mask)

    return zz, yy, xx, mask


from matplotlib import pyplot as plt


def alternate_iter_extend3d(
    zz,
    yy,
    xx,
    active_frames,
    mov,
    verbose=False,
    extend_thresh=0.001,
    max_ext_iters=10,
    extend_z=True,
    max_pix=10000,
    patch_norms=None,
):
    npix = 0
    iter_idx = 0
    mov_act = mov[active_frames]
    lam = n.ones_like(zz, dtype=float) / (len(zz))
    # print("Called alternate_iter_extend_3d")
    while npix < max_pix and iter_idx < max_ext_iters:
        npix = len(yy)
        roi_activity = (mov_act[:, zz, yy, xx] * lam).sum(axis=1)
        roi_activity_normed = roi_activity / n.sqrt((roi_activity**2).sum())
        zz, yy, xx = extend_roi3d(zz, yy, xx, mov.shape[1:], extend_z=extend_z)
        candidate_pix_activity = mov_act[:, zz, yy, xx]
        candidate_norms = patch_norms[zz, yy, xx]
        # print(candidate_pix_activity.shape)
        # print(roi_activity_normed)
        candidate_proj_on_roi = roi_activity_normed @ candidate_pix_activity
        frac_candidate_variance_on_roi = candidate_proj_on_roi / candidate_norms

        incl_pix = frac_candidate_variance_on_roi > extend_thresh

        zz, yy, xx, fracs = [
            vv[incl_pix] for vv in [zz, yy, xx, frac_candidate_variance_on_roi]
        ]
        lam = fracs - extend_thresh
        lam /= n.sqrt((lam**2).sum())

        if incl_pix.sum() == 0:
            if verbose:
                print("Break - no pixels")
            break
        if verbose:
            print(
                "Iter %d, %d/%d pix included" % (iter_idx, incl_pix.sum(), len(incl_pix))
            )
        if not incl_pix.sum() > npix:
            if verbose:
                print("Break - no more growing")
            break
        iter_idx += 1
        # print(candidate_sigma.shape)
        # plt.hist(corrs)
        # plt.show()

        # print(candidate_pix_activity.shape)
        # print("candidates")
        # print(roi_activity.shape)
        # plt.plot(roi_activity)
        # plt.plot(candidate_pix_activity, alpha=0.1)
        # plt.show()
        # lam = lam / n.sum(lam**2) ** 0.5
    return zz, yy, xx, lam


def iter_extend3d(
    zz,
    yy,
    xx,
    active_frames,
    mov,
    verbose=False,
    extend_thresh=0.2,
    max_ext_iters=10,
    extend_z=True,
    max_pix=10000,
    patch_norms=None,
):
    # pr = cProfile.Profile()
    # pr.enable()
    npix = 0
    iter_idx = 0
    mov_act = mov[active_frames].mean(axis=0)
    # lam = n.array([lam0])
    while npix < max_pix and iter_idx < max_ext_iters:
        npix = len(yy)
        zz, yy, xx = extend_roi3d(zz, yy, xx, mov.shape[1:], extend_z=extend_z)
        lam = mov_act[zz, yy, xx]
        incl_pix = lam > max(lam.max() * extend_thresh, 0)

        # print("including %d of %d pixels with max lam")
        if incl_pix.sum() == 0:
            if verbose:
                print("Break - no pixels")
            break
        zz, yy, xx, lam = [vv[incl_pix] for vv in [zz, yy, xx, lam]]
        if verbose:
            print(
                "Iter %d, %d/%d pix included" % (iter_idx, incl_pix.sum(), len(incl_pix))
            )
        if not incl_pix.sum() > npix:
            if verbose:
                print("Break - no more growing")
            break
        iter_idx += 1
    lam = lam / n.sum(lam**2) ** 0.5
    return zz, yy, xx, lam


def extend_roi3d_iter(zz, yy, xx, shape, n_iters=3, extend_z=True):
    for i in range(n_iters):
        zz, yy, xx = extend_roi3d(zz, yy, xx, shape, extend_z=extend_z)
    return zz, yy, xx


def extend_roi_3d_f(zz, yy, xx, shape, extend_z=True):
    pass


def extend_roi3d(zz, yy, xx, shape, extend_z=True):
    n_pix = len(zz)
    coords = [[zz[i], yy[i], xx[i]] for i in range(n_pix)]
    for coord_idx in range(n_pix):
        coord = coords[coord_idx]
        for i in range(3):
            if not extend_z and i == 0:
                continue
            for sign in [-1, 1]:
                v = list(coord)
                v[i] = v[i] + sign
                out_of_bounds = False
                for j in range(len(v)):
                    if v[j] < 0 or v[j] >= shape[j]:
                        out_of_bounds = True
                if not out_of_bounds:
                    coords.append(v)
    zz, yy, xx = n.unique(coords, axis=0).T

    return zz, yy, xx


def extend_helper(vv_roi, vv_ring, extend_v, nv, v_max_extension=None):
    if v_max_extension is None:
        v_max_extension = n.inf
    v_min, v_max = vv_ring.min(), vv_ring.max()
    v_absmin = max(0, vv_roi.min() - v_max_extension, vv_ring.min() - extend_v)
    v_absmax = min(nv, vv_roi.max() + v_max_extension + 1, vv_ring.max() + 1 + extend_v)

    # print(v_absmin)
    return n.arange(v_absmin, v_absmax)


def dilate_mask(mask, extend_by):
    extend_by = tuple(max(0, int(v)) for v in extend_by)
    size = tuple(2 * v + 1 for v in extend_by)
    return maximum_filter(mask.astype(n.uint8), size=size) > 0


def get_neuropil_mask_cell_expansion(
    stat,
    cell_pix,
    min_neuropil_pixels=1000,
    extend_by=(1, 3, 3),
    z_max_extension=5,
    max_np_ext_iters=5,
    return_coords_only=False,
    np_ring_iterations=2,
):
    zz_roi, yy_roi, xx_roi = stat["coords"]
    nz, ny, nx = cell_pix.shape

    zz_ring, yy_ring, xx_ring = extend_roi3d_iter(
        zz_roi, yy_roi, xx_roi, cell_pix.shape, np_ring_iterations
    )
    ring_mask = n.zeros((nz, ny, nx), dtype=bool)
    ring_mask[zz_ring, yy_ring, xx_ring] = True

    grown_mask = ring_mask.copy()
    z_allowed = n.zeros(nz, dtype=bool)
    if z_max_extension is None:
        z_min = 0
        z_max = nz
    else:
        z_min = max(0, int(n.min(zz_roi)) - int(z_max_extension))
        z_max = min(nz, int(n.max(zz_roi)) + int(z_max_extension) + 1)
    z_allowed[z_min:z_max] = True

    neuropil_mask = n.zeros((nz, ny, nx), dtype=bool)
    iter_idx = 0
    n_np_pix = 0
    while n_np_pix < min_neuropil_pixels and iter_idx < max_np_ext_iters:
        grown_mask = dilate_mask(grown_mask, extend_by)
        grown_mask &= z_allowed[:, None, None]
        neuropil_mask = grown_mask & ~ring_mask & ~cell_pix
        n_np_pix = int(neuropil_mask.sum())
        iter_idx += 1

    if return_coords_only:
        zz_np, yy_np, xx_np = n.nonzero(grown_mask)
        return zz_np, yy_np, xx_np, zz_ring, yy_ring, xx_ring

    return n.nonzero(neuropil_mask)


def create_cell_pix(
    stats, shape, lam_percentile=70.0, percentile_filter_shape=(3, 25, 25)
):
    nz, ny, nx = shape
    lam_map = n.zeros((nz, ny, nx))
    roi_map = n.zeros((nz, ny, nx))

    for i, stat in enumerate(stats):
        zc, yc, xc = stat["coords"]
        lam = stat["lam"]
        lam_map[zc, yc, xc] = n.maximum(lam_map[zc, yc, xc], lam)

    if lam_percentile > 0.0:
        filt = percentile_filter(lam_map, percentile=70.0, size=(3, 25, 25))
        cell_pix = ~n.logical_or(lam_map < filt, lam_map == 0)
    else:
        cell_pix = lam_map > 0.0
    return cell_pix


def get_neuropil_mask(
    stat,
    cell_pix,
    min_neuropil_pixels=1000,
    extend_by=(1, 3, 3),
    z_max_extension=5,
    max_np_ext_iters=5,
    return_coords_only=False,
    np_ring_iterations=2,
    neuropil_mask_method="rectangular",
):
    method = str(neuropil_mask_method).lower()
    if method in ("expanding", "cell_expansion", "cell-expansion", "dilate", "dilation"):
        return get_neuropil_mask_cell_expansion(
            stat,
            cell_pix,
            min_neuropil_pixels=min_neuropil_pixels,
            extend_by=extend_by,
            z_max_extension=z_max_extension,
            max_np_ext_iters=max_np_ext_iters,
            return_coords_only=return_coords_only,
            np_ring_iterations=np_ring_iterations,
        )
    if method not in ("rectangular", "rectangle", "default"):
        raise ValueError(
            "Unknown neuropil_mask_method %r. Use 'rectangular' or 'expanding'."
            % neuropil_mask_method
        )

    zz_roi, yy_roi, xx_roi = stat["coords"]
    zz_ring, yy_ring, xx_ring = extend_roi3d_iter(
        zz_roi, yy_roi, xx_roi, cell_pix.shape, np_ring_iterations
    )

    nz, ny, nx = cell_pix.shape

    n_ring = (~cell_pix[zz_ring, yy_ring, xx_ring]).sum()

    zz_np, yy_np, xx_np = zz_ring.copy(), yy_ring.copy(), xx_ring.copy()

    n_np_pix = 0
    iter_idx = 0
    while n_np_pix < min_neuropil_pixels and iter_idx < max_np_ext_iters:
        zs_np = extend_helper(zz_roi, zz_np, extend_by[0], nz, z_max_extension)
        ys_np = extend_helper(yy_roi, yy_np, extend_by[1], ny)
        xs_np = extend_helper(xx_roi, xx_np, extend_by[2], nx)

        zz_np, yy_np, xx_np = n.meshgrid(zs_np, ys_np, xs_np, indexing="ij")
        np_pixs = ~cell_pix[zz_np, yy_np, xx_np]
        n_np_pix = (np_pixs).sum() - n_ring
        # print(n_np_pix)
        # print(zs_np)
        # print(xs_np)

        iter_idx += 1

    if return_coords_only:
        return zz_np, yy_np, xx_np, zz_ring, yy_ring, xx_ring

    else:
        neuropil_mask = n.zeros((nz, ny, nx))
        neuropil_mask[zz_np[np_pixs], yy_np[np_pixs], xx_np[np_pixs]] = True
        neuropil_mask[zz_ring, yy_ring, xx_ring] = False
        pix = n.nonzero(neuropil_mask)

        return pix


def compute_npil_masks_mp_helper(coords, cell_pix_shmem_par, npil_pars, offset):
    shmem, cell_pix = utils.load_shmem(cell_pix_shmem_par)
    npcoords = get_neuropil_mask({"coords": coords}, cell_pix, **npil_pars)
    npcoords_patch = (
        npcoords[0] - offset[0],
        npcoords[1] - offset[1],
        npcoords[2] - offset[2],
    )
    shmem.close()
    return (npcoords, npcoords_patch)


def compute_npil_masks_mp(stats, shape, offset=(0, 0, 0), n_proc=8, npil_pars={}):
    # TODO: parallelize this (EASY)
    # tic = time.time()
    cell_pix = create_cell_pix(stats, shape)
    cell_shmem, cell_shmem_par, cell_pix = utils.create_shmem_from_arr(
        cell_pix, copy=True
    )
    # print(time.time() - tic)
    pool = Pool(n_proc)
    all_np_coords = pool.starmap(
        compute_npil_masks_mp_helper,
        [(stat["coords"], cell_shmem_par, npil_pars, offset) for stat in stats],
    )
    cell_shmem.close()
    cell_shmem.unlink()

    for i, stat in enumerate(stats):
        stat["npcoords"] = all_np_coords[i][0]
        stat["npcoords_patch"] = all_np_coords[i][1]
    return stats


def compute_npil_masks(stats, shape, offset=(0, 0, 0), np_params={}):
    # TODO: parallelize this (EASY)
    cell_pix = create_cell_pix(stats, shape)
    for stat in stats:
        zc, yc, xc = stat["coords"]
        npz, npy, npx = get_neuropil_mask(stat, cell_pix, **np_params)
        stat["npcoords"] = (npz, npy, npx)
        stat["npcoords_patch"] = (npz - offset[0], npy - offset[1], npx - offset[2])
    return stats


def extract_activity_mp(
    mov,
    stats,
    batchsize_frames=500,
    log=default_log,
    offset=None,
    n_frames=None,
    nproc=8,
):
    pass
    # if you run out of memory, reduce batchsize_frames
    # if offset is not None:
    # mov = mov[offset[0][0]:offset[0][1],offset[1][0]:offset[1][1],offset[2][0]:offset[2][1]]

    nz, nt, ny, nx = mov.shape
    if n_frames is None:
        n_frames = nt
    else:
        log("Only extracting %d frames" % n_frames)
        mov = mov[:, :n_frames]
        nt = mov.shape[1]
    print(mov.shape)
    ns = len(stats)
    F_roi = n.zeros((ns, nt))
    F_neu = n.zeros((ns, nt))
    shmem_F_roi, shmem_par_F_roi, F_roi = utils.create_shmem_from_arr(F_roi, copy=True)
    shmem_F_neu, shmem_par_F_neu, F_neu = utils.create_shmem_from_arr(F_neu, copy=True)
    # print(offset)
    n_batches = int(n.ceil(nt / batchsize_frames))
    log("Will extract in %d batches of %d" % (n_batches, batchsize_frames), 3)
    for batch_idx in range(n_batches):
        log("Extracting batch %04d of %04d" % (batch_idx, n_batches), 4)
        start = batch_idx * batchsize_frames
        end = min(nt, start + batchsize_frames)
        mov_batch = mov[:, start:end].compute()
        shmem_batch, shmem_par_batch, mov_batch = utils.create_shmem_from_arr(
            mov_batch, copy=True
        )
        log("Batch size: %.2f GB" % (mov_batch.nbytes / (1024**3),), 4)
        for i in range(ns):
            stat = stats[i]
            zc, yc, xc = stat["coords"]
            npzc, npyc, npxc = stat["npcoords"]

            lam = stat["lam"] / stat["lam"].sum()
            F_roi[i, start:end] = lam @ mov_batch[zc, :, yc, xc]
            F_neu[i, start:end] = mov_batch[npzc, :, npyc, npxc].mean(axis=0)
        shmem_batch.close()
        shmem_batch.unlink()
        del mov_batch

    F_roi_out = F_roi.copy()
    F_neu_out = F_neu.copy()
    shmem_F_roi.close()
    shmem_F_roi.unlink()
    shmem_F_neu.close()
    shmem_F_neu.unlink()
    return F_roi_out, F_neu_out


def extract_helper(mov_shmem):
    pass


def extract_activity(
    mov,
    stats,
    batchsize_frames=500,
    log=default_log,
    offset=None,
    n_frames=None,
    intermediate_save_dir=None,
    mov_shape_tfirst=False,
    npil_to_roi_npix_ratio=None,
    min_npil_npix=0,
    compute_overmerge=False,
    overmerge_max_pix=50,
):
    """Extract fluorescence traces from a movie for each ROI.

    Args:
        compute_overmerge (bool): If True, accumulate per-cell pixel covariance
            matrices across batches and compute overmerge scores at the end.
        overmerge_max_pix (int): Max pixels per cell for overmerge computation.
            Cells with more pixels are subsampled to the top-weighted ones.
            Memory: n_cells * max_pix * max_pix * 4 bytes (fp32).

    Returns:
        F_roi, F_neu: (n_cells, nt) fluorescence and neuropil traces.
        If compute_overmerge: also returns overmerge_scores (n_cells,) array.
    """

    if mov_shape_tfirst:
        nt, nz, ny, nx = mov.shape
    else:
        nz, nt, ny, nx = mov.shape

    if n_frames is None:
        n_frames = nt
    else:
        log("Only extracting %d frames" % n_frames)
        if mov_shape_tfirst:
            mov = mov[:n_frames]
            nt = mov.shape[0]
        else:
            mov = mov[:, :n_frames]
            nt = mov.shape[1]

    ns = len(stats)
    F_roi = n.zeros((ns, nt))
    F_neu = n.zeros((ns, nt))

    # --- Overmerge accumulator setup ---
    om_pix_indices = None
    om_sum = None
    om_ssq = None
    om_n = None
    om_p = None  # actual p per cell (may be < overmerge_max_pix)

    if compute_overmerge:
        p = overmerge_max_pix
        # Check projected size: ns * p * p * 4 bytes
        projected_bytes = ns * p * p * 4
        projected_gb = projected_bytes / (1024**3)
        if projected_gb > 20:
            old_p = p
            # Scale p down so total stays under 20 GB
            p = int(n.sqrt(20 * 1024**3 / (ns * 4)))
            p = max(p, 5)  # absolute minimum
            log("WARNING: Overmerge accumulator would be %.1f GB at p=%d. "
                "Auto-scaling to p=%d (%.1f GB)" % (projected_gb, old_p, p, ns * p * p * 4 / 1024**3))
            overmerge_max_pix = p

        log("Overmerge: accumulating covariance for %d cells, p=%d (%.1f MB)" %
            (ns, overmerge_max_pix, ns * overmerge_max_pix * overmerge_max_pix * 4 / (1024**2)), 2)

        # Pre-compute which pixels to use per cell (top by lam weight)
        om_pix_indices = []
        om_p = n.zeros(ns, dtype=int)
        for i, stat in enumerate(stats):
            if stat is None:
                om_pix_indices.append(None)
                continue
            lam = stat.get("lam", n.array([]))
            npix = len(lam)
            if npix < 6:  # need at least 6 pixels for 5 PCs
                om_pix_indices.append(None)
                continue
            if npix <= overmerge_max_pix:
                om_pix_indices.append(n.arange(npix))
                om_p[i] = npix
            else:
                top_idx = n.argsort(lam)[-overmerge_max_pix:]
                om_pix_indices.append(top_idx)
                om_p[i] = overmerge_max_pix

        # Allocate accumulators — use memmap if large, else in-memory
        if projected_gb > 2:
            log("Overmerge: using memmap accumulators (%.1f GB)" % projected_gb, 2)
            tmpdir = intermediate_save_dir or "."
            om_sum_path = os.path.join(tmpdir, "_om_sum_tmp.npy")
            om_ssq_path = os.path.join(tmpdir, "_om_ssq_tmp.npy")
            # Create files
            om_sum = n.lib.format.open_memmap(
                om_sum_path, mode='w+', dtype=n.float32,
                shape=(ns, overmerge_max_pix))
            om_ssq = n.lib.format.open_memmap(
                om_ssq_path, mode='w+', dtype=n.float32,
                shape=(ns, overmerge_max_pix, overmerge_max_pix))
        else:
            om_sum = n.zeros((ns, overmerge_max_pix), dtype=n.float32)
            om_ssq = n.zeros((ns, overmerge_max_pix, overmerge_max_pix), dtype=n.float32)
        om_n = n.zeros(ns, dtype=n.int64)

    # --- Main extraction loop ---
    n_batches = int(n.ceil(nt / batchsize_frames))
    batch_save_interval = 100
    log("Will extract in %d batches of %d" % (n_batches, batchsize_frames), 3)
    if intermediate_save_dir is not None:
        log("Saving intermediate results to %s" % intermediate_save_dir)

    for batch_idx in range(n_batches):
        log("Extracting batch %04d of %04d" % (batch_idx, n_batches), 4)
        start = batch_idx * batchsize_frames
        end = min(nt, start + batchsize_frames)
        try:
            if mov_shape_tfirst:
                mov_batch = mov[start:end].swapaxes(0, 1).compute()
            else:
                mov_batch = mov[:, start:end].compute()
        except Exception:
            log("NOT A DASK ARRAY!", 3)
            mov_batch = mov[:, start:end]
        log("Batch size: %d GB" % (mov_batch.nbytes / (1024**3),), 4)

        for i in range(ns):
            stat = stats[i]
            if stat is None:
                continue

            zc, yc, xc = stat["coords"]
            npzc, npyc, npxc = stat["npcoords"]
            if npil_to_roi_npix_ratio is not None:
                npix_roi = len(zc)
                npix_npil = len(npzc)
                if npix_npil > npix_roi * npil_to_roi_npix_ratio:
                    n_sample = max(min_npil_npix, int(npix_roi * npil_to_roi_npix_ratio))
                    if npix_npil < n_sample:
                        n_sample = npix_npil

                    sample_idxs = n.random.choice(
                        n.arange(npix_npil), size=n_sample, replace=False
                    )
                    npzc = npzc[sample_idxs]
                    npyc = npyc[sample_idxs]
                    npxc = npxc[sample_idxs]

            # Pixel raster: (n_pix, nt_batch)
            fpixs = mov_batch[zc, :, yc, xc]
            lam = stat["lam"] / stat["lam"].sum()
            F_roi[i, start:end] = lam @ fpixs
            F_neu[i, start:end] = mov_batch[npzc, :, npyc, npxc].mean(axis=0)

            # Overmerge: accumulate covariance on subsampled pixels
            if compute_overmerge and om_pix_indices[i] is not None:
                pidx = om_pix_indices[i]
                pi = om_p[i]
                pix = fpixs[pidx, :].astype(n.float32)  # (pi, nt_batch)
                om_sum[i, :pi] += pix.sum(axis=1)
                om_ssq[i, :pi, :pi] += pix @ pix.T  # (pi, pi)
                om_n[i] += pix.shape[1]

        if (
            (intermediate_save_dir is not None)
            and (batch_idx > 0)
            and (batch_idx % batch_save_interval == 0)
        ):
            log(
                "Batch %d: Saving intermediate results to %s"
                % (batch_idx, intermediate_save_dir)
            )
            n.save(os.path.join(intermediate_save_dir, "F.npy"), F_roi)
            n.save(os.path.join(intermediate_save_dir, "Fneu.npy"), F_neu)

    # --- Compute overmerge scores from accumulators ---
    if compute_overmerge:
        log("Computing overmerge scores from accumulated covariance matrices...", 2)
        overmerge_scores = _finalize_overmerge(om_sum, om_ssq, om_n, om_p, ns, overmerge_max_pix)

        # Clean up memmap temp files
        if isinstance(om_sum, n.memmap):
            del om_sum, om_ssq
            tmpdir = intermediate_save_dir or "."
            for fname in ["_om_sum_tmp.npy", "_om_ssq_tmp.npy"]:
                fpath = os.path.join(tmpdir, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)

        return F_roi, F_neu, overmerge_scores

    return F_roi, F_neu


def _finalize_overmerge(om_sum, om_ssq, om_n, om_p, ns, max_pix, n_pcs=5):
    """Compute overmerge scores from accumulated sufficient statistics.

    For each cell, computes the covariance matrix from the running sum and
    sum-of-squares, then extracts the top eigenvalues to derive the
    overmerge score.
    """
    from .quality_metrics import overmerge_score_calc

    overmerge_scores = n.full(ns, n.nan, dtype=n.float32)

    for i in range(ns):
        ni = om_n[i]
        pi = om_p[i]
        if ni < n_pcs + 1 or pi < n_pcs + 1:
            continue

        mean_i = om_sum[i, :pi] / ni  # (pi,)
        # cov = E[X X^T] - E[X] E[X]^T
        cov_i = om_ssq[i, :pi, :pi] / ni - n.outer(mean_i, mean_i)

        # Eigenvalues of the covariance matrix (ascending order)
        try:
            eigvals = n.linalg.eigvalsh(cov_i)
        except n.linalg.LinAlgError:
            continue

        # Take top n_pcs eigenvalues (descending)
        top_eigvals = eigvals[-n_pcs:][::-1]

        if len(top_eigvals) < n_pcs:
            padded = n.zeros(n_pcs)
            padded[:len(top_eigvals)] = top_eigvals
            top_eigvals = padded

        # Compute overmerge score
        score = overmerge_score_calc(top_eigvals[n.newaxis, :])[0]
        overmerge_scores[i] = score

    return overmerge_scores


def prune_overlapping_cells(stats, dist_thresh=5, lam_overlap_thresh=0.5):
    meds = n.array([s["med"] for s in stats])
    lams = [n.array(s["lam"]) for s in stats]
    # med_patchs = n.array([s['med_patch'] for s in stats])
    coords = [n.array(s["coords"]).T for s in stats]

    dm = distance_matrix(meds, meds, threshold=10000)
    dm[n.tril_indices(dm.shape[0], 1)] = dist_thresh + 1

    pairs = n.array(n.where((dm < dist_thresh)))

    pair_fracs = []
    max_lam = []
    for pair in pairs.T:
        p0, p1 = pair
        c0 = coords[p0]
        c1 = coords[p1]

        t0 = [(tuple(c)) for c in c0]
        t1 = [(tuple(c)) for c in c1]

        intersect = []
        lam_intersect_0 = 0
        lam_intersect_1 = 0
        for idx0, t in enumerate(t0):
            if t in t1:
                idx1 = t1.index(t)
                lam_intersect_0 += lams[p0][idx0]
                lam_intersect_1 += lams[p1][idx1]
                intersect.append(t)
        n_intersect = len(intersect)
        n0 = len(c0)
        n1 = len(c1)
        nx = min(n0, n1)
        frac_intersect = n_intersect / nx
        pair_fracs.append(frac_intersect)
        max_lam.append(
            max(lam_intersect_0 / lams[p0].sum(), lam_intersect_1 / lams[p1].sum())
        )
    max_lam = n.array(max_lam)

    overlap_pairs_flag = max_lam >= lam_overlap_thresh
    overlap_pairs = pairs[:, overlap_pairs_flag]
    duplicate_cells = n.zeros(meds.shape[0], dtype=bool)
    for overlap_pair in overlap_pairs.T:
        op0, op1 = overlap_pair
        lamsum0 = lams[op0].sum()
        lamsum1 = lams[op1].sum()
        if lamsum0 > lamsum1:
            duplicate_cells[op1] = True
        else:
            duplicate_cells[op0] = True
    new_stats = []
    for idx, stat in enumerate(stats):
        if not duplicate_cells[idx]:
            new_stats.append(stat)
    return new_stats, duplicate_cells


def thresh_mask_corr_map(im3d, thresh_window_size_pix=51, corrmap_thresh_pct=50):
    window_size = (thresh_window_size_pix // 2) * 2 + 1
    nz, ny, nx = im3d.shape
    out = []
    for plane_idx in range(nz):
        img = im3d[plane_idx].copy()
        minval = n.percentile(img.flatten(), corrmap_thresh_pct)
        img[img < minval] = minval
        thresh_local = threshold_local(img, block_size=window_size)
        masked_img = img.copy()
        masked_img[img < thresh_local] = img.min()
        out.append(masked_img - minval)
    out = n.array(out)
    return out
