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