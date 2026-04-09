import os 
import numpy as n
from multiprocessing import Pool
from .utils import default_log, binned_robust_regression
from . import utils
from scipy.ndimage import uniform_filter
from .extension import find_top_n_rois, filter_rois, save_checkpoint, save_final_results, log_cell_addition
import time 



def segment_rois(msub, vmap, n_proc_detect = 8, peak_thresh = 1.0, activity_thresh=None,
                 vox_snr_thresh=1.0, roi_ext_iterations=10, roi_dilations_per_iter = 3, max_pix = 10000, min_pix = 4, use_power_iter_v1 = False,multi_source = True,
                 roi_power_iterations = 5, roi_min_active_frames = 50, roi_recompute_active_frames_interval=None, ext_subtract_iters=3,
                 vox_snr_mp_correction=False,
                 savepath =None, patch_idx = -1, offset = (0,0,0), max_iter = 1e6, log=default_log, **kwargs):

    stats = []
    offset = n.array(offset)
    # log("Loading movie patch to shared memory", 3)
    shmem_msub, shmem_par_msub, msub = utils.create_shmem_from_arr(msub, copy=True)
    msub_vars = ((msub**2).sum(axis=0))
    # log("Loaded", 3)

    n_iters = int(max_iter // n_proc_detect)
    worker_idxs = n.arange(n_proc_detect)
    roi_idx = 0
    nt, nz, ny, nx = msub.shape

    t0 = time.time()
    with Pool(n_proc_detect) as p:
        prev_n_rois = 0
        for iter_idx in range(n_iters):
            n_rois = len(stats)
            potential_rois = find_top_n_rois(vmap, n_rois=n_proc_detect)
            potential_rois = filter_rois(potential_rois, peak_thresh)

            if not potential_rois:
                log(f"Extracted {n_rois} ROIs; no more candidates found above peak {peak_thresh:.2f}, stopping.", 0)
                break
            log(f"Iteration {iter_idx}: segmenting {len(potential_rois)} candidate ROIs (total so far: {n_rois})", 2)
            roi_idxs = n.arange(len(potential_rois)) + roi_idx + 1

            new_rois = p.starmap(
                segment_roi,
                [
                    (shmem_par_msub, msub_vars, vmap, potential_rois[widx], activity_thresh, max_pix, min_pix, use_power_iter_v1,
                     roi_ext_iterations, roi_dilations_per_iter, roi_power_iterations, roi_min_active_frames, vox_snr_thresh,
                     multi_source, roi_recompute_active_frames_interval,
                     offset, False, t0, roi_idxs[widx], worker_idxs[widx], patch_idx, vox_snr_mp_correction)
                    for widx in range(len(potential_rois))
                ]
            )            
            # print('return')
            # print(len(new_rois))
            add_segmented_rois(new_rois, stats, msub, vmap, log, ext_subtract_iters=ext_subtract_iters)

            # Update variances for voxels affected by the subtracted cells
            for stat in new_rois:
                if stat is None:
                    continue
                zz_s = stat['coords'][0] - stat['offset'][0]
                yy_s = stat['coords'][1] - stat['offset'][1]
                xx_s = stat['coords'][2] - stat['offset'][2]
                msub_vars[zz_s, yy_s, xx_s] = (msub[:, zz_s, yy_s, xx_s] ** 2).sum(axis=0)
            roi_idx = len(stats)
            n_rois = len(stats)
            n_rois_iter = n_rois - prev_n_rois
            prev_n_rois = n_rois
            log(f"Iter {iter_idx:04d}: added {n_rois_iter} ROIs, total {n_rois}", 2)
            if n_rois_iter == 0:
                log("No ROIs added this iteration - ending extraction", 2)
                break

            if iter_idx % 50 == 0 and savepath is not None:
                save_checkpoint(savepath, stats, log)

    shmem_msub.close()
    shmem_msub.unlink()
    log(f"Found {roi_idx} cells in {iter_idx+1} iterations")
    # print(stats[0].keys())
    save_final_results(savepath, stats, log)
    return stats


def zero_roi_in_vmap(
    vmap, zz, yy, xx, ext_subtract_iters=1
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

    zzx, yyx, xxx = extend_roi3d(zz, yy, xx, vmap.shape, extend_z=True)

    for i in range(ext_subtract_iters):
        zzx, yyx, xxx = extend_roi3d(
            zzx, yyx, xxx, (nz, ny, nx), extend_z=False
        )
    vmap[zzx, yyx, xxx] = 0

def add_segmented_rois(new_rois, stats, msub, vmap, log, ext_subtract_iters=3, regress_vmap=False):
    # print("adding rois", len(stats))
    for stat in new_rois:
        if stat is None:
            continue
            
        zz,yy,xx = stat['coords'][0] - stat['offset'][0], stat['coords'][1] - stat['offset'][1], stat['coords'][2] - stat['offset'][2]
        med = stat['med'] - stat['offset']
        f1_u_active = stat['f1_u']
        active = stat['active_frames']
        lam = stat['lam']

        msub[active[:, None], zz[None, :], yy[None, :], xx[None, :]] -= n.outer(f1_u_active, lam)

        if regress_vmap: 
            vmap[zz,yy,xx] -= lam * stat['vmap_slope'] + stat['vmap_int']
        else:
            # print("Zeroing ", med)
            # print("Before:", vmap[med[0], med[1], med[2]])
            zero_roi_in_vmap(vmap, zz, yy, xx, ext_subtract_iters=ext_subtract_iters)
            # print("After:", vmap[med[0], med[1], med[2]])
        stats.append(stat)
        # print(stat['active_frames'])
# 
        log_cell_addition(log, stat, len(stats)+1)
    print(vmap[0,215-120,279-242])


def segment_roi(msub, variances, vmap, roi_init, activity_thresh = 5,max_pix=10000,min_pix=4,use_power_iter_v1 = False,
                roi_ext_iterations = 1, roi_dilations_per_iter = 3, n_power_iter = 3, min_frames = 50,vox_snr_thresh=1.0,
                multi_source=True,recompute_active_frames_interval=None, offset=(0,0,0),
                debug = False, t_start = 0, roi_idx = -1, worker_idx = -1, patch_idx = -1,
                vox_snr_mp_correction=False):
    '''
    assume a model where p(x) = v1 f1(t) + v2 f2(t) + noise
    '''

    shmem = False
    if type(msub) == dict:
        shmem = True
        msub_sh, msub = utils.load_shmem(msub)

    t0 = time.time()
    t00 = time.time()
    med, zz, yy, xx, v1, peak = roi_init
    seed_zz, seed_yy, seed_xx = zz.copy(), yy.copy(), xx.copy()
    v1 /= (v1**2).sum()**0.5

    if msub.shape[0] < min_frames:
        min_frames = msub.shape[0]
        # default_log('')
    if debug: print("Extracting ROI at z,y,x = %d,%d,%d with peak %.2f and %d frames" % (med[0],med[1],med[2], peak, msub.shape[0]))
    if debug: print("Time to start: %.5f ms" % (1000*(time.time() - t0))); t0 = time.time()
    F0 = msub[:, zz, yy, xx]  # nt x nvox
    npix = len(zz)
    npixs = [npix]
    f1_init = F0 @ v1
    if debug: print("Time to compute initial f1: %.5f ms" % (1000*(time.time() - t0))); t0 = time.time()
    
    active_frames, active_frame_idxs = compute_active_frames(f1_init, activity_thresh, min_frames)
    f1 = f1_init[active_frame_idxs]
    f1 /= (f1**2).sum()**0.5
    if debug: print("Time to compute active frames: %.5f ms" % (1000*(time.time() - t0))); t0 = time.time()
    if debug: print(f"Time to init: {1000*(time.time() - t00):.5f} ms")

    extend_iter = 0
    while extend_iter < roi_ext_iterations: 
        extend_iter += 1

        t0 = time.time()   
        cz, cy, cx = zz, yy, xx
        if extend_iter > 1:
            for i in range(roi_dilations_per_iter):
                cz, cy, cx = extend_roi3d(cz, cy, cx, vmap.shape, extend_z=True)
        else:
            cz, cy, cx = extend_roi3d(cz, cy, cx, vmap.shape, extend_z=True)
        
        Fc = msub[:, cz, cy, cx][active_frame_idxs]  # nt x nvox
        v1h_u = f1 @ Fc
        v1h = v1h_u / (v1h_u**2).sum()**0.5
        # print(v1h_u)
        # print(variances[cz,cy,cx])
        if use_power_iter_v1:
            v1h_u, v1h, f1_u, f1 = power_iteration(Fc, n_power_iter=n_power_iter, f0=f1)

        # if multi_source:
        Fc_res = Fc - n.outer(f1, v1h_u)  # nt x nvox
        v2h_u, v2h, f2_u, f2 = power_iteration(Fc_res, n_power_iter=n_power_iter)
        contamination_factor =  (v1h @ v2h)
        if multi_source:
            v1_u = v1h_u - contamination_factor * v2h_u
        else:
            v1_u = v1h_u 

        if vox_snr_mp_correction:
            T_active = Fc.shape[0]
            K = Fc.shape[1]
            gamma = K / T_active
            active_var = (Fc**2).sum(axis=0)
            noise_var = (active_var - v1_u**2) / T_active
            noise_var = n.maximum(noise_var, 1e-10)
            term1 = v1_u**2 / T_active - (1 + gamma) * noise_var
            term2 = n.sqrt(n.maximum(term1**2 - 4 * gamma * noise_var**2, 0))
            alpha_sq = (T_active / 2) * (term1 + term2)
            alpha_sq = n.maximum(alpha_sq, 0)
            vox_snrs = alpha_sq / (T_active * noise_var)
        else:
            vox_snrs = v1_u**2 / (variances[cz,cy,cx] - v1_u**2)
        include = vox_snrs > vox_snr_thresh
        include[:len(seed_zz)] = True

        v1_u = v1_u[include]
        vox_snrs = vox_snrs[include]
        zz,yy,xx = cz[include], cy[include], cx[include]
        # make sure we always include the seed voxels
        v1h_u = v1h_u[include]
        v2_u = v2h_u[include] #if multi_source else None

        v1 = v1_u / (v1_u**2).sum()**0.5
        f1_u = Fc[:,include] @ v1
        f1 = f1_u / (f1_u**2).sum()**0.5
        f2 = f2_u / (f2_u**2).sum()**0.5 #if multi_source else None

        f1f2 = f1 @ f2 #if multi_source else 0
        if debug: print("Time for extend iteration %d: %.5f ms" % (extend_iter, time.time() - t0))
        t0 = time.time()

        if recompute_active_frames_interval is not None and (extend_iter + 1) % recompute_active_frames_interval == 0:
            f1_u_f = msub[:, zz, yy, xx] @ v1
            active_frames, active_frame_idxs = compute_active_frames(f1_u_f, activity_thresh, min_frames)
            f1_u = f1_u_f[active_frames]
            f1 = f1_u / (f1_u**2).sum()**0.5
        npix = len(zz)
        npixs.append(npix)

        if npix > max_pix:
            if debug: print(f"Reached max pixels {max_pix}, stopping extension.")
            break
    # print(npixs)
    if npix < min_pix:
        return None
    lam = v1_u / (v1_u**2).sum()**0.5
    vmap_masked = vmap[zz,yy,xx]
    vmap_slope, vmap_int = utils.binned_robust_regression(lam,  vmap_masked, x_bins=8, pct=90, below=False, return_points=False)

    extraction_time = time.time() - t00
    elapsed_time = time.time() - t_start
    if debug: print("Total time for extract_roi: %.5f ms" % (1000*(time.time() - t00)))
    if shmem: msub_sh.close()

    stat = {
        # basic outputs
        'idx' : roi_idx, # unique ROI index
        'med' : (med[0] + offset[0], med[1] + offset[1], med[2] + offset[2]), # starting voxel
        'coords' : (zz + offset[0], yy + offset[1], xx + offset[2]), # coordinates of voxels
        'lam' : lam, # normalized value of ROI mask per voxel

        # internal (algorithm-related) values
        'v1_u': v1_u,
        'v1h_u' : v1h_u,
        'v2_u': v2_u,
        'f1_u': f1_u,
        'f2_u': f2_u,
        'f1f2': f1f2,
        'vox_snrs': vox_snrs,
        # 'f1_u_full' : msub[:, zz, yy, xx] @ lam,
        'peak_val': peak,
        'active_frames': active_frame_idxs,
        'contamination_factor': contamination_factor,
        'npixs': npixs,
        'threshold' : activity_thresh,

        # correlation map-related
        'vmap_masked' : vmap_masked,
        'vmap_slope' : vmap_slope,
        'vmap_int' : vmap_int,

        # multi-patch extraction info
        'worker_idx': worker_idx,
        'patch_idx': patch_idx,
        'offset' : offset,

        # timing
        'extraction_time': extraction_time,
        'elapsed_time': elapsed_time,
    }


    return stat

    


def compute_active_frames(f1_u, activity_thresh=5, min_frames=50):
    active_frames = f1_u > activity_thresh
    nf = active_frames.sum()
    if nf < min_frames:
        additional_frames = f1_u.argsort()[-min_frames:]
        active_frames[additional_frames] = True
    nf = active_frames.sum()
    active_frame_idxs = n.where(active_frames)[0]
    return active_frames, active_frame_idxs


def power_iteration(Fc, n_power_iter=3, f0=None):
    nt, nv = Fc.shape
    if f0 is None:
        f0 = n.random.normal(size=nt)
    else:
        f0 = f0 / (f0**2).sum()**0.5
    v_u = Fc.T @ f0
    v = v_u / (v_u**2).sum()**0.5
    for power_iter in range(n_power_iter):
        f_u = Fc @ v_u
        f = f_u / (f_u**2).sum()**0.5
        v_u = f @ Fc
        v = v_u / (v_u**2).sum()**0.5

    # make them all mean positive
    if v.sum() < 0:
        v_u = -v_u
        v = -v
        f_u = -f_u
        f = -f
    
    return v_u, v, f_u, f


def extend_roi3d(zz, yy, xx, shape, extend_z=True):
    """
    Vectorized ROI dilation by 6-neighborhood (or 4-neighborhood if extend_z=False).
    Original coordinates are guaranteed to be at the front of the output.
    """
    base = n.stack([zz, yy, xx], axis=1)  # (n_pix, 3)
    
    # Offsets for neighbors only (exclude [0,0,0])
    offsets = n.array(
        [[0, 0, -1], [0, 0, 1],
         [0, -1, 0], [0, 1, 0],
         [-1, 0, 0], [1, 0, 0]], dtype=int
    )
    if not extend_z:
        offsets = offsets[:4]  # drop z-shifts

    # Generate neighbor coordinates
    coords_extended = base[:, None, :] + offsets[None, :, :]  # (n_pix, n_off, 3)
    coords_extended = coords_extended.reshape(-1, 3)           # (n_pix * n_off, 3)

    # Filter in-bounds
    nz, ny, nx = shape
    m = (
        (coords_extended[:, 0] >= 0) & (coords_extended[:, 0] < nz) &
        (coords_extended[:, 1] >= 0) & (coords_extended[:, 1] < ny) &
        (coords_extended[:, 2] >= 0) & (coords_extended[:, 2] < nx)
    )
    coords_extended = coords_extended[m]
    
    # Remove duplicates within extended coords
    coords_extended = n.unique(coords_extended, axis=0)
    
    # Remove any extended coords that are already in the original set
    # Create a set of tuples for fast lookup
    original_set = set(map(tuple, base))
    mask = n.array([tuple(coord) not in original_set for coord in coords_extended])
    coords_extended = coords_extended[mask]
    
    # Concatenate: originals first, then new neighbors
    coords_all = n.concatenate([base, coords_extended], axis=0)
    
    return coords_all[:, 0], coords_all[:, 1], coords_all[:, 2]

def extend_roi3d_old(zz, yy, xx, shape, extend_z=True):
    """
    Vectorized ROI dilation by 6-neighborhood (or 4-neighborhood if extend_z=False).
    """
    base = n.stack([zz, yy, xx], axis=1)  # (n_pix, 3)
    offsets = n.array(
        [[0, 0, 0],   # keep originals
         [0, 0, -1], [0, 0, 1],
         [0, -1, 0], [0, 1, 0],
         [-1, 0, 0], [1, 0, 0]], dtype=int
    )
    if not extend_z:
        offsets = offsets[[0, 1, 2, 3, 4]]  # drop z-shifts

    coords = base[:, None, :] + offsets[None, :, :]      # (n_pix, n_off, 3)
    coords = coords.reshape(-1, 3)                       # (n_pix * n_off, 3)

    # in-bounds mask
    nz, ny, nx = shape
    m = (
        (coords[:, 0] >= 0) & (coords[:, 0] < nz) &
        (coords[:, 1] >= 0) & (coords[:, 1] < ny) &
        (coords[:, 2] >= 0) & (coords[:, 2] < nx)
    )
    coords = coords[m]
    coords = n.unique(coords, axis=0)
    return coords[:, 0], coords[:, 1], coords[:, 2]


def filter_movie(mov, spatial_filt):
    """Apply spatial filtering to each frame of the movie.

    Args:
        mov (np.ndarray): 4D array of shape (nt, nz, ny, nx)
        spatial_filt (tuple): Tuple of three integers specifying the size of the uniform filter in (z, y, x)

    Returns:
        np.ndarray: Filtered movie of the same shape as input.
    """
    nt, nz, ny, nx = mov.shape
    filtered_mov = n.empty_like(mov)
    for t in range(nt):
        filtered_mov[t] = uniform_filter(mov[t], size=(1,spatial_filt,spatial_filt), mode='reflect')
    return filtered_mov