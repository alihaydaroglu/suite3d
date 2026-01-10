import numpy as n
import os
import re
import copy
from scipy.ndimage import shift as ndi_shift
import imreg_dft


def get_cell_centroids(coords, lams):
    """
    Compute the weighted centroids of cells from Suite3D

    Args:
        coords (list): list of cell coords, output of Suite3D
        lams (list): List of pixel weights per cell, output of Suite3D

    Returns:
        centroids: (n_cells, n_dim) array of weighted centroids
    """
    n_cells = len(coords)
    ndim = len(coords[0])
    centroids = n.zeros((n_cells, ndim))

    # loop over all cells
    for cell_idx in range(n_cells):
        coord = coords[cell_idx]
        lam = lams[cell_idx]
        lam_sum = lam.sum()
        # for each coordinate, take the weighted average
        # weighted by the 'lam' values
        centroids[cell_idx] = [(coord[i] * lam).sum() / lam_sum for i in range(ndim)]

    return centroids


def filter_coords_by_nvox(stats, min_nvox=5, sample='random', seed=0, key_prefix='', keep_small_rois = False, add_none = False):
    '''
    Filter coordinates based on minimum number of voxels
    If sample is 'random', randomly sample min_nvox voxels
    If sample is 'top', take the top min_nvox voxels by lam value
    '''
    fstats = []
    n.random.seed(seed)
    n_valid = 0
    for roi_idx, stat in enumerate(stats):
        stat = copy.deepcopy(stat)
        zc, yc, xc = stat[key_prefix + 'coords']
        lam = stat[key_prefix + 'lam'].copy()
        n_vox = len(lam)
        if n_vox <= min_nvox:
            if add_none:
                stat['fidx'] = n.array([], dtype=int)
                stat['fcoords'] = (n.array([], dtype=int), n.array([], dtype=int), n.array([], dtype=int))
                stat['flam'] = n.array([], dtype=float)
                fstats.append(stat)
            elif keep_small_rois:
                stat['fidx'] = n.arange(n_vox)
                stat['fcoords'] = (zc, yc, xc)
                stat['flam'] = lam
                fstats.append(stat)
            continue
        if sample == 'random':
            keep_idx = n.random.choice(n_vox, size=min_nvox, replace=False)
        elif sample == 'top':
            lam_sort_idx = n.argsort(lam)[::-1]
            keep_idx = lam_sort_idx[:min_nvox]
        else:
            lam_sort_idx = n.argsort(lam)[::-1]
            keep_idx = lam_sort_idx[:min_nvox]
        n_valid += 1
        keep_idx = keep_idx[n.argsort(lam[keep_idx])[::-1]]
        stat['fidx'] = keep_idx
        stat['fcoords'] = (zc[keep_idx], yc[keep_idx], xc[keep_idx])
        stat['flam'] = lam[keep_idx]
        fstats.append(stat)

    if not keep_small_rois:
        print(f"Filtered from {len(stats)} to {n_valid} stats with at least {min_nvox} voxels")
    else:
        print(f"Kept all {len(stats)} stats, of which {n_valid} have at least {min_nvox} voxels")
    return fstats

def filter_coords(stats, lam_sum_frac = 0.75, prefix='f'):
    '''
    Filter coordinates based on cumulative sum of lam values
    Saves the coordinates that contribute a certain fraction of a cells weights
    into fcoords and flam. Also saves the indices into fidx.
    '''

    mean_nvox_before = n.mean([len(stat['lam']) for stat in stats])
    for roi_idx, stat in enumerate(stats):
        zc, yc, xc = stat['coords']
        lam = stat['lam'].copy()
        # lam = lam / lam.sum()
        # sort lam in descending order
        lam_sort_idx = n.argsort(lam)[::-1]
        lam_sorted = lam[lam_sort_idx]
        lam_cumsum = n.cumsum(lam_sorted)
        # find the index where the cumulative sum exceeds lam_sum_frac
        thresh_idx = n.where(lam_cumsum >= (lam_sum_frac * lam.sum()))[0][0]
        # get the threshold value
        lam_thresh = lam_sorted[thresh_idx]
        # filter coords and lam
        keep_idx = n.where(lam >= lam_thresh)[0]
        keep_idx = keep_idx[n.argsort(lam[keep_idx])[::-1]]
        # stat['flag'] = lam >= lam_thresh
        stat[f'{prefix}idx'] = keep_idx
        stat[f'{prefix}coords'] = (zc[keep_idx], yc[keep_idx], xc[keep_idx])
        stat[f'{prefix}lam'] = lam[keep_idx]
    mean_nvox_after = n.mean([len(stat[f'{prefix}lam']) for stat in stats])
    print(f"Mean nvox before: {mean_nvox_before:.2f}, after: {mean_nvox_after:.2f}")
    return stats

def shift_coords(stats, shifts):
    '''
    Shift coordinates in the stats list based on the provided shifts (from align_volumes_2d)
    to match two analyses of the same recording (only 2D shifts, so it wont work well across recordings)
    '''
    for roi_idx, stat in enumerate(stats):
        if 'raw_coords' in stat:
            print("Already shifted coords")
            break
        zc, yc, xc = stat['coords']
        lam = stat['lam']
        stat['raw_coords'] = (n.copy(zc), n.copy(yc), n.copy(xc))   
        stat['raw_lam'] = n.copy(lam)

        for idx in range(len(shifts)):
            shift_y, shift_x = shifts[idx]
            yc[zc == idx] += shift_y
            xc[zc == idx] += shift_x
        stat['coords'] = (zc, yc, xc)

    return stats 

def align_volumes_2d(reference, volume):
    '''
    compute the 2d shifts to align each slice of volume to reference
    '''

    nz, ny, nx = reference.shape
    nz, ny_vol , nx_vol = volume.shape
    # import ndimage shift

    vol_shifts = []
    volume_shifted = []

    for iz in range(nz):
        ref_slice = reference[iz]
        ref_slice /= ref_slice.max()
        vol_slice = volume[iz]
        vol_slice /= vol_slice.max()

        # pad or crop caiman slice to match suite3d slice size, pad only on right and bottom
        if ny_vol < ny:
            pad_y = ny - ny_vol
            vol_slice = n.pad(vol_slice, ((0, pad_y), (0, 0)), mode='constant')
        elif ny_vol > ny:
            vol_slice = vol_slice[:ny, :]
        if nx_vol < nx:
            pad_x = nx - nx_vol
            vol_slice = n.pad(vol_slice, ((0, 0), (0, pad_x)), mode='constant')
        elif nx_vol > nx:
            vol_slice = vol_slice[:, :nx]
        # Register images
        sim = imreg_dft.translation(ref_slice, vol_slice)
        shift = - n.round(-sim['tvec'])
        vol_shifts.append(shift)
        # the y and x shifts, in the order: (y_shift, x_shift)
        vol_slice_shifted = ndi_shift(vol_slice, shift=shift)   
        volume_shifted.append(vol_slice_shifted)

    volume_shifted = n.array(volume_shifted)
    vol_shifts = n.array(vol_shifts).astype(int)

    return vol_shifts, volume_shifted


def correlate(array, vector):
    """
    compute the correlation of each element in array with given vector

    Args:
        array (ndarray): n_cells, nt
        vector (ndarray): nt

    Returns:
        correlation of each row of the array with the vector, size n_cells
    """
    vector = n.squeeze(vector)
    squeeze_output = False
    if len(array.shape) == 1:
        array = array[n.newaxis, :]
        squeeze_output = True
    array = array - array.mean(axis=1, keepdims=True)
    vector = vector - vector.mean(axis=0)
    cov = (array * vector[n.newaxis]).sum(axis=1)
    var_arr = n.sqrt((array**2).sum(axis=1))
    var_vec = n.sqrt((vector**2).sum(axis=0))
    # print(vector.shape)
    # print(cov.shape, var_arr.shape, var_vec.shape)

    out = cov / (var_arr * var_vec)
    if squeeze_output:
        out = float(n.squeeze(out))

    return out


def normalize_columns(X):
    """
    Normalize columns by their standard deviation.
    """
    return X / X.std(axis=0)


def compute_voxel_projections_and_correlations(fpixs, loading0, loading1, npix_clus=15):
    """
    Compute voxel-based projections and correlations for two loading vectors.
    """
    load_diff = (loading1) - (loading0)
    load_sort = n.argsort(load_diff)

    pc1_pix = fpixs[:, load_sort[:npix_clus]].mean(axis=1)
    pc2_pix = fpixs[:, load_sort[-npix_clus:]].mean(axis=1)

    pc1_corr = correlate(fpixs.T, pc1_pix)
    pc2_corr = correlate(fpixs.T, pc2_pix)
    corr_diff = pc2_corr - pc1_corr

    return load_diff, load_sort, pc1_pix, pc2_pix, pc1_corr, pc2_corr, corr_diff

def make_roi_box(zc, yc, xc, lam, box_half_sizes=(3, 10, 10), volumes=None, centroid=None, lam_norm=True):
    """Build a 3D mask box around an ROI centroid, optionally extracting
    aligned sub-boxes from provided reference volumes.

    Args:
        zc, yc, xc (array-like): Integer voxel coordinates of the ROI (same length).
        lam (array-like): Weights per voxel (same length as coords).
        box_half_sizes (tuple): Half sizes (dz, dy, dx) of the box around centroid.
        volumes (list/tuple/dict, optional): Collection of 3D volumes with shape
            (Z, Y, X). If dict, keys are used as names; if list/tuple, names
            auto-generated as 'vol0', 'vol1', ...

    Returns:
        dict: {'centroid', 'mask', 'box_shape', <volume boxes...>}
              Each volume box is stored under its name.
    """
    zc = n.asarray(zc)
    yc = n.asarray(yc)
    xc = n.asarray(xc)
    lam = n.asarray(lam)

    if not (zc.size == yc.size == xc.size == lam.size):
        raise ValueError("zc, yc, xc, lam must have same length")

    dz, dy, dx = box_half_sizes
    box_shape = (2*dz + 1, 2*dy + 1, 2*dx + 1)

    # Centroid (weighted) and normalized weights
    if centroid is None:
        centroid_float = get_cell_centroids([(zc, yc, xc)], [lam])
        centroid = n.round(centroid_float).astype(int)[0]
    if lam_norm: 
        lamp = lam / lam.max() if lam.max() != 0 else lam
    else:
        lamp = lam

    # Initialize mask box with NaNs
    maskbox = n.zeros(box_shape) * n.nan

    # Populate sparse points
    zp = zc - centroid[0]
    yp = yc - centroid[1]
    xp = xc - centroid[2]
    for i in range(zp.size):
        zpi, ypi, xpi = int(zp[i]), int(yp[i]), int(xp[i])
        zi, yi, xi = dz + zpi, dy + ypi, dx + xpi
        if 0 <= zi < box_shape[0] and 0 <= yi < box_shape[1] and 0 <= xi < box_shape[2]:
            maskbox[zi, yi, xi] = lamp[i]

    out = {
        'centroid': centroid,
        'mask': maskbox,
        'box_shape': box_shape,
    }

    if volumes is None:
        return out

    # Normalize volumes input to iterable of (name, volume)
    if isinstance(volumes, dict):
        named_vols = list(volumes.items())
    else:
        named_vols = [(f'vol{i}', v) for i, v in enumerate(volumes)]

    for name, vol in named_vols:
        vol = n.asarray(vol)
        if vol.ndim != 3:
            raise ValueError(f"Volume '{name}' must be 3D")

        # Bounds within volume
        z_start = max(0, centroid[0] - dz)
        z_end = min(vol.shape[0], centroid[0] + dz + 1)
        y_start = max(0, centroid[1] - dy)
        y_end = min(vol.shape[1], centroid[1] + dy + 1)
        x_start = max(0, centroid[2] - dx)
        x_end = min(vol.shape[2], centroid[2] + dx + 1)

        # Offsets if centroid near boundary
        z_off = max(0, dz - centroid[0])
        y_off = max(0, dy - centroid[1])
        x_off = max(0, dx - centroid[2])

        subvol = vol[z_start:z_end, y_start:y_end, x_start:x_end]
        vbox = n.zeros(box_shape) * n.nan
        vbox[z_off:z_off + subvol.shape[0],
             y_off:y_off + subvol.shape[1],
             x_off:x_off + subvol.shape[2]] = subvol
        out[name] = vbox

    return out


def make_roi_boxes(zc, yc, xc, lam, pc1_corr, pc2_corr, corr_diff, s3d_mean, s3d_max, cmn_vmap,
                   box_half_sizes=(3, 10, 10)):
    """
    Build 3D boxes around ROI centroid for mask, correlations, and images.
    """
    print('xxx')
    dz, dy, dx = box_half_sizes
    box_shape = (2*dz+1, 2*dy+1, 2*dx+1)

    # centroid and normalized weights
    centroid_float = get_cell_centroids([(zc, yc, xc)], [lam])
    centroid = n.round(centroid_float).astype(int)[0]
    lamp = lam / lam.max() if lam.max() != 0 else lam

    # initialize boxes
    maskbox = n.zeros(box_shape) * n.nan
    load0box = n.zeros(box_shape) * n.nan
    load1box = n.zeros(box_shape) * n.nan
    loadDbox = n.zeros(box_shape) * n.nan
    meanbox = n.zeros(box_shape) * n.nan
    maxbox = n.zeros(box_shape) * n.nan
    cmn_vmapbox = n.zeros(box_shape) * n.nan

    # place sparse points for mask and correlation diffs
    zp, yp, xp = zc - centroid[0], yc - centroid[1], xc - centroid[2]
    for i in range(len(zp)):
        zpi, ypi, xpi = int(zp[i]), int(yp[i]), int(xp[i])
        zi, yi, xi = dz + zpi, dy + ypi, dx + xpi
        if 0 <= zi < box_shape[0] and 0 <= yi < box_shape[1] and 0 <= xi < box_shape[2]:
            maskbox[zi, yi, xi] = lamp[i]
            load0box[zi, yi, xi] = pc1_corr[i]
            load1box[zi, yi, xi] = pc2_corr[i]
            loadDbox[zi, yi, xi] = corr_diff[i]

    # extract dense slices from volumes with bounds checks
    z_start, z_end = max(0, centroid[0]-dz), min(s3d_mean.shape[0], centroid[0]+dz+1)
    y_start, y_end = max(0, centroid[1]-dy), min(s3d_mean.shape[1], centroid[1]+dy+1)
    x_start, x_end = max(0, centroid[2]-dx), min(s3d_mean.shape[2], centroid[2]+dx+1)

    z_off = max(0, dz - centroid[0])
    y_off = max(0, dy - centroid[1])
    x_off = max(0, dx - centroid[2])

    emean = s3d_mean[z_start:z_end, y_start:y_end, x_start:x_end]
    emax = s3d_max[z_start:z_end, y_start:y_end, x_start:x_end]
    evmap = cmn_vmap[z_start:z_end, y_start:y_end, x_start:x_end]

    meanbox[z_off:z_off+emean.shape[0], y_off:y_off+emean.shape[1], x_off:x_off+emean.shape[2]] = emean
    maxbox[z_off:z_off+emax.shape[0], y_off:y_off+emax.shape[1], x_off:x_off+emax.shape[2]] = emax
    cmn_vmapbox[z_off:z_off+evmap.shape[0], y_off:y_off+evmap.shape[1], x_off:x_off+evmap.shape[2]] = evmap

    # normalize to start at zero for display
    if not n.isnan(n.nanmin(maxbox)):
        maxbox -= n.nanmin(maxbox)
    if not n.isnan(n.nanmin(meanbox)):
        meanbox -= n.nanmin(meanbox)

    return {
        'centroid': centroid,
        'mask': maskbox,
        'pc1_corr': load0box,
        'pc2_corr': load1box,
        'corr_diff': loadDbox,
        'mean': meanbox,
        'max': maxbox,
        'cmn_vmap': cmn_vmapbox,
        'box_shape': box_shape,
    }


def get_caiman_dirs(root_dir, param_names = ('K', 'merge', 'rval')):
    

    # List all directories in savedir
    all_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    # Regex to extract K, merge, rval from directory names

    # make a general regex expression (one below is for specific 3 params) which extracts anything after the param name between _ 
    # base it on the param_names input
    pattern = re.compile(r"_".join([f"{name}_([0-9.]+)" for name in param_names]))
    caiman_params = []
    caiman_dirs = []
    for d in all_dirs:
        m = pattern.match(d)
        if m:
            K, merge, rval = m.groups()
            key = (K, merge, rval)
            caiman_params.append((K, merge, rval))
            caiman_dirs.append(os.path.join(root_dir, d,))
    return caiman_dirs, caiman_params


def corr_mat(arr, nan_diag=False, dtype=n.float32, eps=1e-6):
    """
    compute correlation matrix of a data matrix. Similar to n.corrcoef

    Args:
        arr (ndarray): n_cells, n_timepoints

    Returns:
        corr: n_cells, n_cells
    """
    arr = arr.astype(dtype)
    arr = arr - arr.mean(axis=1, keepdims=True)

    cov = arr @ arr.T

    var = (arr**2).sum(axis=1)

    corr = cov / (n.sqrt(var[:, n.newaxis] @ var[n.newaxis]) + eps)

    if nan_diag:
        n.fill_diagonal(corr, n.nan)
    return corr



def dist_mat(meds, type="euclidean"):
    """
    Get pairwise distances betweel all pairs of cells along all dimensions

    Args:
        meds (ndarray): (n_cells, ndim) array of the centroid of all cells
        type (str) : 'euclidean', 'manhattan' or 'vector'

    Returns:
        dists: (n_cells, n_cells) array of pairwise distances.
               if type == 'vector', then (n_cells, n_cells, ndim)
    """
    n_cells, n_dims = meds.shape
    dists = n.stack([meds[:, i][:, n.newaxis] - meds[:, i][n.newaxis] for i in range(n_dims)], axis=-1)

    if type == "vector":
        return dists
    elif type == "euclidean":
        n.square(dists, dists)
        return n.sqrt(dists.sum(axis=-1))
    elif type == "manhattan":
        n.abs(dists, dists)
        return dists.sum(axis=-1)


def flatten_lower_tri(matrix, k=-1):
    """
    return a flattened version of the lower triangular elements of matrix, excluding the the diagonal by default

    Args:
        matrix (ndarray): square matrix
        k (int, optional): Offset, -1 excludes diagonal, 0 includes diagonal. Defaults to -1.

    Returns:
        ndarray: flattened matrix of size (I think) nx * (nx - 1) / 2?
    """
    trilidx = n.tril_indices_from(matrix, -1)
    flat_matrix = matrix[trilidx]
    return flat_matrix




def bin_pairwise_percentiles(pairwise_dists, pairwise_corrs, percentiles, dist_bins):
    """
    Bin 1D pairwise distances and compute percentiles of the corresponding
    1D pairwise correlation values within each distance bin.

    Inputs
    ------
    pairwise_dists : ndarray, shape (n_pairs,)
        Distances for each pair.
    pairwise_corrs : ndarray, shape (n_pairs,)
        Correlation (or any scalar metric) per pair, aligned with pairwise_dists.
    percentiles : tuple of float
        Percentiles to compute (e.g., (5, 25, 50, 75, 95)).
    dist_bins : ndarray, shape (n_bins+1,)
        Monotonic array of bin edges.

    Behavior
    --------
        - Bin definition uses half-open intervals [left, right) for all bins.
            Values equal to the final rightmost edge are excluded.
    - NaN values in either distances or correlations are ignored.
    - Empty bins yield NaN for all requested percentiles.

    Returns
    -------
    bin_centers : ndarray, shape (n_bins,)
        Centers of the provided bins.
    pct_dict : dict[float, ndarray]
        Mapping from percentile value to an array of shape (n_bins,) with the
        percentile of the distribution within each bin.
    """
    d = n.asarray(pairwise_dists).ravel()
    v = n.asarray(pairwise_corrs).ravel()
    edges = n.asarray(dist_bins).ravel()

    if d.shape != v.shape:
        raise ValueError("pairwise_dists and pairwise_corrs must have the same shape")
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("dist_bins must be a 1D array of length >= 2 (bin edges)")
    if not n.all(n.isfinite(edges)):
        raise ValueError("dist_bins must be finite")
    if n.any(n.diff(edges) < 0):
        raise ValueError("dist_bins must be monotonically non-decreasing")

    nbins = edges.size - 1
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    # mask out NaNs in either input
    valid = (~n.isnan(d)) & (~n.isnan(v))
    d = d[valid]
    v = v[valid]

    # Assign bins with right edge inclusive behavior on the last bin
    # Equivalent to idx = searchsorted(edges, d, 'right') - 1
    idx = n.searchsorted(edges, d, side="right") - 1

    # Keep only values that fall within [0, nbins-1]
    inrange = (idx >= 0) & (idx < nbins)
    idx = idx[inrange]
    v = v[inrange]

    # Prepare outputs
    percentiles = tuple(percentiles)
    pct_dict = {float(p): n.full(nbins, n.nan, dtype=float) for p in percentiles}

    if idx.size == 0:
        return bin_centers, pct_dict

    # Group by bin index efficiently
    order = n.argsort(idx, kind="mergesort")  # stable
    idx_sorted = idx[order]
    v_sorted = v[order]
    uniq_bins, starts, counts = n.unique(idx_sorted, return_index=True, return_counts=True)

    for ub, start, count in zip(uniq_bins.tolist(), starts.tolist(), counts.tolist()):
        if ub < 0 or ub >= nbins or count <= 0:
            continue
        sl = slice(start, start + count)
        vals = v_sorted[sl]
        for p in percentiles:
            pct_dict[float(p)][ub] = n.nanpercentile(vals, float(p))

    return bin_centers, pct_dict



def get_pair_idx(indices, n_dim=None, matrix=None, k=-1):
    """
    Map flat lower-triangle indices back to (row, col) pairs.

    Args:
        indices (int | iterable[int]): Integer index or iterable of indices into flatten_lower_tri output.
        n_dim (int, optional): Size of the (square) matrix. Required if matrix not provided.
        matrix (ndarray, optional): Matrix whose shape is used to infer n_dim.
        k (int, optional): Offset passed to tril_indices (same meaning as numpy.tril_indices). Defaults to -1.

    Returns:
        ndarray: Shape (n_pairs, 2) with (row, col) coordinates.
    """
    if matrix is not None:
        n_dim = matrix.shape[0]
    if n_dim is None:
        raise ValueError("Provide n_dim or matrix.")
    indices = n.atleast_1d(indices).astype(int)
    if indices.ndim != 1:
        raise ValueError("indices must be 1D or scalar.")
    tril_i, tril_j = n.tril_indices(n_dim, k=k)
    max_valid = tril_i.size - 1
    if (indices < 0).any() or (indices > max_valid).any():
        raise IndexError(f"indices out of bounds for lower triangle of size {tril_i.size}.")
    return n.column_stack((tril_i[indices], tril_j[indices]))


def zscore(x, nax=0, m=None, std=None, return_params=False, auto_reshape=True, undo=False):
    """zscore a given axis of an n-dimensional array based on given or computed parameters.
       If you have an array of shape x,y,z and nax=1, the activity will be average over all
       x and z, so the mean and std will have shape 1,y,1.

    Args:
        x (ndarray): ndim array
        nax (list or int, optional): Axes to *not* average over, typically the neuron axis. Defaults to 0.
        m (ndarray, optional): mean. Defaults to computing from x.
        std (ndarray, optional): std. Defaults to computing from x.
        return_params (bool, optional): Return m and std in a tuple. Defaults to False.
        auto_reshape (bool, optional): Automatically fix the shapes of m and std. Defaults to True.
    """
    # x = n.squeeze(x)
    ndim = len(x.shape)
    if ndim == 1:
        nax = [-1]
    nax = n.array(nax).astype(int)
    axes_to_reduce = n.array([i if i not in nax else n.nan for i in range(ndim)])
    axes_to_reduce = tuple(n.array(axes_to_reduce)[~n.isnan(axes_to_reduce)].astype(int))
    if m is None:
        m = x.mean(axis=axes_to_reduce, keepdims=True)
    if std is None:
        std = x.std(axis=axes_to_reduce, keepdims=True)

    std += 1e-6

    if auto_reshape:
        param_shape = n.ones(ndim).astype(int)
        param_shape[nax] = n.array(x.shape)[nax]

        # if they are a scalar don't reshape
        if n.array(m).size > 1:
            m = m.reshape(*param_shape)
        if n.array(std).size > 1:
            std = std.reshape(*param_shape)

    if not undo:
        xz = (x - m) / std
    else:
        xz = (x * std) + m
    if return_params:
        return xz, (m, std)
    else:
        return xz
    

def compute_weighted_overlap(coords1, lam1, coords2, lam2):
    """
    Compute weighted overlap between two sets of coordinates and weights.

    Args:
        coords1 (tuple of ndarray): (zc1, yc1, xc1) coordinates for set 1.
        lam1 (ndarray): Weights for set 1. (squared sum = 1)
        coords2 (tuple of ndarray): (zc2, yc2, xc2) coordinates for set 2.
        lam2 (ndarray): Weights for set 2. (squared sum = 1)
    Returns:
        float: Weighted overlap value.
    """
    zc1, yc1, xc1 = coords1
    zc2, yc2, xc2 = coords2

    # Hash map of coordinates -> summed weight for first set (handles duplicates gracefully)
    d1 = {}
    for i in range(len(lam1)):
        key = (int(zc1[i]), int(yc1[i]), int(xc1[i]))
        d1[key] = d1.get(key, 0.0) + float(lam1[i])

    # Accumulate overlap using second set
    overlap = 0.0
    for j in range(len(lam2)):
        key = (int(zc2[j]), int(yc2[j]), int(xc2[j]))
        w1 = d1.get(key)
        if w1 is not None:
            overlap += w1 * float(lam2[j])

    return overlap


def compute_weighted_overlap_fast(coords1, lam1, coords2, lam2):
    """Fast weighted overlap between two coordinate sets.

    Uses a single hash-map pass (O(n1 + n2)) rather than the
    naive double loop (O(n1 * n2)). Handles duplicate coordinates
    by summing their weights before multiplying with the second set.

    Args:
        coords1 (tuple(ndarray, ndarray, ndarray)): (zc1, yc1, xc1) integer coords.
        lam1 (ndarray): Weights for set 1.
        coords2 (tuple(ndarray, ndarray, ndarray)): (zc2, yc2, xc2) integer coords.
        lam2 (ndarray): Weights for set 2.
    Returns:
        float: Weighted overlap value.
    """
    zc1, yc1, xc1 = coords1
    zc2, yc2, xc2 = coords2
    lam1 = n.asarray(lam1)
    lam2 = n.asarray(lam2)

    if not (len(zc1) == len(yc1) == len(xc1) == len(lam1)):
        raise ValueError("coords1 arrays and lam1 must have same length")
    if not (len(zc2) == len(yc2) == len(xc2) == len(lam2)):
        raise ValueError("coords2 arrays and lam2 must have same length")

    d1 = {}
    for i in range(len(lam1)):
        key = (int(zc1[i]), int(yc1[i]), int(xc1[i]))
        d1[key] = d1.get(key, 0.0) + float(lam1[i])

    overlap = 0.0
    for j in range(len(lam2)):
        key = (int(zc2[j]), int(yc2[j]), int(xc2[j]))
        w1 = d1.get(key)
        if w1 is not None:
            overlap += w1 * float(lam2[j])

    return overlap