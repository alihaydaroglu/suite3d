import numpy as n
from . import quality_utils as qu


def volume_quality(volume, pct_high = 99.99, pct_low = 25.0):
    '''
    return some heuristics about the signal level in the mean volume provided

    Args:
        volume (ndarray): nz,ny,nx - averaged movie over time
        pct_high (float, optional): high percentile to take as the peak of 'signal' . Defaults to 99.99.
        pct_low (float, optional): low percentile to take as 'background'. Defaults to 25.0.

    Returns:
        metrics: dictionary
    '''
    sig = n.percentile(volume, pct_high, axis=(1,2))
    bg = n.percentile(volume, pct_low, axis=(1,2))


    metrics = {
        'signal_range':  sig - bg,
        'signal_to_background_ratio' : sig / bg,
        'mean_fluorescence' : volume.mean(axis=(1,2)),
        'volume_std'  : volume.std(axis=(1,2)),
    }
    return metrics

def shot_noise_pct(fs, frate_hz):
    '''
    compute the theoretical shot noise percentage in a timeseries 
    assumes GCamP6s, using equation from Pachitariu

    Args:
        fs (ndarray): npix, nt - timeseries
        frate_hz (float): frame rate

    Returns:
        noise_level: array of percentage noise for each pixel
    '''
    df = n.diff(fs, axis=1)
    dff = df / fs.mean(axis=1,keepdims=True)
    abs_d_dff = n.abs(n.diff(dff,axis=1))
    noise_level = n.nanmedian(abs_d_dff, axis=1)
    noise_level = noise_level / frate_hz

    return noise_level

def choose_top_pix(vol, pct = 98):
    nz, ny, nx = vol.shape
    pcts = n.array([n.percentile(vol[i].flatten(), pct)  for i in range(nz)])
    n_top_pix = int((100-pct) * ny * nx / 100 - 2)
    top_pix = n.array([vol[i] >= pcts[i] for i in range(nz)])
    return top_pix

def compute_metrics_for_movie(mov, frate_hz, top_pix=None):
    nz, nt, ny, nx = mov.shape
    vol = mov.mean(axis=1)
    metrics = volume_quality(vol)
    if top_pix is None:
        top_pix = choose_top_pix(vol)

    noises = []
    npix = (top_pix.sum(axis=(1,2))).min()
    # print(npix)
    for i in range(nz):
        noises.append(shot_noise_pct(mov[i][:,top_pix[i]][:,:npix].reshape(nt,-1).T, frate_hz))
        # print(noises[-1].shape)
    noise_levels = n.array(noises)

    metrics['noise_levels'] = noise_levels

    return vol, metrics

    
def duplicate_score_pairs_slow(F, meds_um, near_thresh = 20.0, use_z=False):
    '''
    Slow implementation that computes the full correlation matrix.

    F: n_cells x n_time - fluorescence traces
    meds_um: n_cells x 3 - zyx positions in um
    near_thresh: float - threshold distance in um to consider 'nearby'
    '''

    corrmat = qu.corr_mat(F)
    corrmat_flat = qu.flatten_lower_tri(corrmat)

    if use_z:
        dists = qu.dist_mat_3d(meds_um)
    else:
        dists = qu.dist_mat(meds_um[:,1:])
    dists_flat = qu.flatten_lower_tri(dists)

    near_mask = dists_flat < near_thresh
    near_idxs = n.where(near_mask)[0]
    near_pair_idxs = qu.get_pair_idx(near_idxs, matrix=dists)
    near_corrs = corrmat_flat[near_idxs]

    return near_pair_idxs, near_corrs


def duplicate_score_pairs(F, meds_um, near_thresh = 20.0, use_z=False, chunk_size=50000):
    '''
    Compute correlation scores only for nearby cell pairs.

    This avoids building the full n_cells x n_cells correlation matrix
    and instead computes correlations only for distance-selected pairs
    in a vectorized, batched way.

    F: n_cells x n_time - fluorescence traces
    meds_um: n_cells x 3 - zyx positions in um
    near_thresh: float - threshold distance in um to consider 'nearby'
    chunk_size: int - number of pairs per batch for correlation compute
    '''

    # distances and nearby pair indices
    if use_z:
        dists = qu.dist_mat_3d(meds_um)
    else:
        dists = qu.dist_mat(meds_um[:,1:])
    dists_flat = qu.flatten_lower_tri(dists)

    near_mask = dists_flat < near_thresh
    near_idxs = n.where(near_mask)[0]
    if near_idxs.size == 0:
        return n.empty((0, 2), dtype=int), n.empty((0,), dtype=float), dists_flat[near_idxs]

    near_pair_idxs = qu.get_pair_idx(near_idxs, matrix=dists)

    # normalize traces once: z-score across time per cell
    F = n.asarray(F, dtype=n.float32)
    F_mean = F.mean(axis=1, keepdims=True)
    F_std = F.std(axis=1, keepdims=True)
    # avoid division by zero for flat traces
    F_std[F_std == 0] = 1.0
    Fz = (F - F_mean) / F_std

    T = F.shape[1]
    K = near_pair_idxs.shape[0]
    near_corrs = n.empty(K, dtype=n.float32)

    # batched computation of correlations for selected pairs
    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        pairs_batch = near_pair_idxs[start:end]
        pi = pairs_batch[:, 0]
        pj = pairs_batch[:, 1]
        Fi = Fz[pi]
        Fj = Fz[pj]
        # elementwise product summed over time dimension -> dot per pair
        numer = n.einsum('kt,kt->k', Fi, Fj)
        near_corrs[start:end] = numer / (T - 1)

    return near_pair_idxs, near_corrs, dists_flat[near_idxs]


def footprint_distance(coords1, lam1, coords2, lam2, p: int = 1):
    """Compute a distance between two weighted spatial footprints.

    The footprints are represented as lists of voxel coordinates with
    associated non-negative weights. This uses a Wasserstein (Earth
    Mover's) distance when the POT (Python Optimal Transport) library
    is available, and falls back to a simple centroid distance
    otherwise.

    Args:
        coords1 (ndarray): (n1, d) voxel coordinates for footprint 1.
        lam1 (ndarray): (n1,) non-negative weights for footprint 1.
        coords2 (ndarray): (n2, d) voxel coordinates for footprint 2.
        lam2 (ndarray): (n2,) non-negative weights for footprint 2.
        p (int, optional): Order of the Wasserstein distance. Use 1 for
            classic EMD. Defaults to 1.

    Returns:
        float: Distance between the two footprints, 0 iff they are
            identical (up to floating point) when POT is installed.
    """
    coords1 = n.asarray(coords1, dtype=float).T
    coords2 = n.asarray(coords2, dtype=float).T
    lam1 = n.asarray(lam1, dtype=float).ravel()
    lam2 = n.asarray(lam2, dtype=float).ravel()

    if coords1.ndim != 2 or coords2.ndim != 2:
        raise ValueError("coords1 and coords2 must be 2D arrays of shape (n_points, dim)")
    if coords1.shape[0] != lam1.shape[0] or coords2.shape[0] != lam2.shape[0]:
        raise ValueError("coords and lambda arrays must have matching lengths")

    # handle empty or zero-mass footprints
    mass1 = lam1.sum()
    mass2 = lam2.sum()
    if mass1 <= 0 and mass2 <= 0:
        return 0.0
    if mass1 <= 0 or mass2 <= 0:
        # one empty, one non-empty -> distance is size of non-empty footprint
        # here we just return the norm of its weighted spread as a simple proxy
        non_empty_coords = coords1 if mass1 > 0 else coords2
        non_empty_lam = lam1 if mass1 > 0 else lam2
        centroid = (non_empty_coords * (non_empty_lam[:, None] / non_empty_lam.sum())).sum(axis=0)
        diffs = non_empty_coords - centroid
        return float(n.sqrt((n.sum(non_empty_lam * n.sum(diffs**2, axis=1)) / non_empty_lam.sum())))

    # normalize to probability masses
    lam1 = lam1 / mass1
    lam2 = lam2 / mass2

    # try to use POT for true Wasserstein distance
    try:
        import ot  # type: ignore
    except ImportError:
        # Fallback: distance between weighted centroids (still a metric,
        # but does not capture full shape differences).
        c1 = (coords1 * lam1[:, None]).sum(axis=0)
        c2 = (coords2 * lam2[:, None]).sum(axis=0)
        return float(n.linalg.norm(c1 - c2))

    C = ot.dist(coords1, coords2, metric="euclidean")
    if p != 1:
        C = C ** p

    # ot.emd2 returns the optimal transport cost
    emd_cost = ot.emd2(lam1, lam2, C)
    if p == 1:
        return float(emd_cost)
    return float(emd_cost ** (1.0 / p))


# =============================================================================
# Overmerge metrics (PCA-based)
# =============================================================================

def overmerge_score_calc(pca_expvars, noise_pc0=3):
    '''Compute overmerge score from PCA explained variances.

    Score near 0 = single source, near 1 = likely two merged sources.

    Args:
        pca_expvars (ndarray): (n_rois, n_pcs) explained variance per component.
        noise_pc0 (int): First PC index considered noise floor.

    Returns:
        ndarray: (n_rois,) overmerge scores.
    '''
    expvar_denoised = pca_expvars - pca_expvars[:, noise_pc0:].mean(axis=-1, keepdims=True)
    denom = expvar_denoised[:, 0] + expvar_denoised[:, 1]
    denom = n.where(denom == 0, 1e-10, denom)
    om_score = 1 - (expvar_denoised[:, 0] - expvar_denoised[:, 1]) / denom
    return om_score


def _pca_expvars_svd(data, n_pcs):
    '''Compute PCA explained variances using numpy SVD (no sklearn needed).

    Args:
        data (ndarray): (n_samples, n_features) centered or uncentered.
        n_pcs (int): Number of components.

    Returns:
        ndarray: (n_pcs,) explained variances (not fractions).
    '''
    data = data - data.mean(axis=0, keepdims=True)
    nt = data.shape[0]
    # Use economy SVD: only compute min(nt, n_features) singular values
    s = n.linalg.svd(data, full_matrices=False, compute_uv=False)
    # Explained variance = s^2 / (n_samples - 1)
    expvars = (s ** 2) / (nt - 1)
    return expvars[:n_pcs]


def overmerge_scores_for_rois(stats, mov, n_pcs=5, key_prefix=''):
    '''Compute per-ROI overmerge scores using voxel-level PCA.

    For each ROI, extracts pixel timeseries, computes SVD, and derives
    the overmerge score from explained variance ratios.

    Args:
        stats (list[dict]): ROI statistics with coords and lam keys.
        mov (ndarray): (nt, nz, ny, nx) movie data.
        n_pcs (int): Number of PCA components.
        key_prefix (str): Prefix for coord/lam keys in stats.

    Returns:
        pca_expvars (ndarray): (n_rois, n_pcs) explained variances.
        overmerge_scores (ndarray): (n_rois,) overmerge scores.
    '''
    nc = len(stats)
    pca_expvars = n.full((nc, n_pcs), n.nan)

    for roi_idx, stat in enumerate(stats):
        zc, yc, xc = stat[key_prefix + 'coords']
        lam = stat[key_prefix + 'lam']
        if len(lam) < n_pcs + 1:
            continue

        fpixs = mov[:, zc, yc, xc]  # (nt, n_pix)
        n_components = min(n_pcs, fpixs.shape[1], fpixs.shape[0])
        if n_components < n_pcs:
            continue

        expvars = _pca_expvars_svd(fpixs, n_pcs)
        pca_expvars[roi_idx, :len(expvars)] = expvars

    valid = ~n.isnan(pca_expvars[:, 0])
    om_scores = n.full(nc, n.nan)
    if valid.any():
        om_scores[valid] = overmerge_score_calc(pca_expvars[valid])

    return pca_expvars, om_scores


def overmerge_scores_batched(stats, mov, n_pcs=5, key_prefix='', batch_size=500):
    '''Batched version of overmerge_scores_for_rois for memory efficiency.

    Same interface but processes ROIs in batches to limit memory use
    when many ROIs have large footprints.
    '''
    nc = len(stats)
    pca_expvars = n.full((nc, n_pcs), n.nan)

    for start in range(0, nc, batch_size):
        end = min(start + batch_size, nc)
        for roi_idx in range(start, end):
            stat = stats[roi_idx]
            zc, yc, xc = stat[key_prefix + 'coords']
            lam = stat[key_prefix + 'lam']
            if len(lam) < n_pcs + 1:
                continue

            fpixs = mov[:, zc, yc, xc]
            n_components = min(n_pcs, fpixs.shape[1], fpixs.shape[0])
            if n_components < n_pcs:
                continue

            expvars = _pca_expvars_svd(fpixs, n_pcs)
            pca_expvars[roi_idx, :len(expvars)] = expvars

    valid = ~n.isnan(pca_expvars[:, 0])
    om_scores = n.full(nc, n.nan)
    if valid.any():
        om_scores[valid] = overmerge_score_calc(pca_expvars[valid])

    return pca_expvars, om_scores


# =============================================================================
# Comprehensive ROI quality summary
# =============================================================================

def compute_roi_metrics(stats, F=None, meds_um=None, voxel_size_um=None,
                        mov=None, frate_hz=None,
                        near_thresh=20.0, n_pcs=5, min_npix=0):
    '''Compute a comprehensive set of quality metrics for detected ROIs.

    This is the main entry point for quality analysis. It computes:
    - Size distribution (n_voxels per ROI)
    - Duplication scores (trace correlation of nearby pairs)
    - Overmerge scores (PCA-based, if mov is provided)
    - Shot noise levels (if F and frate_hz provided)

    Args:
        stats (list[dict]): ROI statistics.
        F (ndarray, optional): (n_rois, nt) fluorescence traces.
        meds_um (ndarray, optional): (n_rois, 3) centroid positions in um.
            If None, computed from stats using voxel_size_um.
        voxel_size_um (tuple, optional): (z, y, x) voxel size for converting
            pixel coordinates to microns.
        mov (ndarray, optional): (nt, nz, ny, nx) movie for overmerge computation.
        frate_hz (float, optional): Frame rate for shot noise computation.
        near_thresh (float): Distance threshold for duplicate detection (um).
        n_pcs (int): PCA components for overmerge.
        min_npix (int): Minimum voxel count to include an ROI.

    Returns:
        dict with keys:
            n_rois: int
            n_voxels: (n_rois,) array
            meds_um: (n_rois, 3) array
            iscell_size: (n_rois,) bool — True if n_voxels >= min_npix
            duplicate_pairs: (n_pairs, 2) int array
            duplicate_corrs: (n_pairs,) float array
            duplicate_dists: (n_pairs,) float array
            overmerge_scores: (n_rois,) float array (NaN if mov not provided)
            pca_expvars: (n_rois, n_pcs) array
            shot_noise: (n_rois,) float array (NaN if F/frate not provided)
    '''
    nc = len(stats)
    n_voxels = n.array([len(s.get('lam', [])) for s in stats])

    # Compute centroids in um
    if meds_um is None:
        if voxel_size_um is None:
            voxel_size_um = (1.0, 1.0, 1.0)
        vox = n.array(voxel_size_um)
        meds_um = n.zeros((nc, 3))
        for i, s in enumerate(stats):
            if 'med' in s:
                meds_um[i] = n.array(s['med']) * vox
            elif 'coords' in s and len(s['lam']) > 0:
                coords = s['coords']
                lam = s['lam']
                lam_norm = lam / lam.sum()
                meds_um[i] = n.array([
                    (coords[0] * lam_norm).sum(),
                    (coords[1] * lam_norm).sum(),
                    (coords[2] * lam_norm).sum(),
                ]) * vox

    results = {
        'n_rois': nc,
        'n_voxels': n_voxels,
        'meds_um': meds_um,
        'iscell_size': n_voxels >= min_npix,
    }

    # Duplication scores
    if F is not None and meds_um is not None:
        dup_pairs, dup_corrs, dup_dists = duplicate_score_pairs(
            F, meds_um, near_thresh=near_thresh
        )
        results['duplicate_pairs'] = dup_pairs
        results['duplicate_corrs'] = dup_corrs
        results['duplicate_dists'] = dup_dists
    else:
        results['duplicate_pairs'] = n.empty((0, 2), dtype=int)
        results['duplicate_corrs'] = n.empty(0)
        results['duplicate_dists'] = n.empty(0)

    # Overmerge scores
    if mov is not None:
        pca_expvars, om_scores = overmerge_scores_batched(stats, mov, n_pcs=n_pcs)
        results['overmerge_scores'] = om_scores
        results['pca_expvars'] = pca_expvars
    else:
        results['overmerge_scores'] = n.full(nc, n.nan)
        results['pca_expvars'] = n.full((nc, n_pcs), n.nan)

    # Shot noise
    if F is not None and frate_hz is not None:
        results['shot_noise'] = shot_noise_pct(F, frate_hz)
    else:
        results['shot_noise'] = n.full(nc, n.nan)

    return results


def summarize_sweep_metrics(sweep_results, stat_key='stats', F_key='F',
                            voxel_size_um=None, frate_hz=None,
                            near_thresh=20.0, min_npix=0, mov=None):
    '''Compute quality metrics for each combination in a parameter sweep.

    Args:
        sweep_results (list[dict]): Each element has stat_key and optionally F_key.
        voxel_size_um, frate_hz, near_thresh, min_npix: passed to compute_roi_metrics.
        mov (ndarray, optional): Movie for overmerge computation.

    Returns:
        list[dict]: Per-combination metrics from compute_roi_metrics.
    '''
    all_metrics = []
    for i, res in enumerate(sweep_results):
        stats = res.get(stat_key, [])
        F = res.get(F_key)
        metrics = compute_roi_metrics(
            stats, F=F, voxel_size_um=voxel_size_um, frate_hz=frate_hz,
            near_thresh=near_thresh, min_npix=min_npix, mov=mov,
        )
        all_metrics.append(metrics)
    return all_metrics