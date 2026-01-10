import numpy as np


def est_npilcoeff_pctregression_cell(F, Fneu, pct=5, neu_bins=20, return_points=False):
    """
    Estimate neuropil coefficients using percentile regression.
    
    Parameters:
    F: array of fluorescence signals
    Fneu: array of neuropil signals  
    pct: percentile threshold for regression (default 5)
    neu_bins: number of bins for neuropil signal (default 20)
    
    Returns:
    coefficient: estimated neuropil coefficient
    intercept: estimated intercept
    """
    
    # Flatten arrays if needed
    xs = Fneu
    ys = F
    
    # Create bins for neuropil signal
    xs_bins = np.linspace(xs.min(), xs.max(), neu_bins + 1)
    
    # Digitize to assign each point to a bin
    bin_indices = np.digitize(xs, xs_bins) - 1
    bin_indices = np.clip(bin_indices, 0, neu_bins - 1)  # Handle edge cases
    
    # Collect points below percentile threshold for each bin
    npil_est_xs = []
    npil_est_ys = []
    
    for i in range(neu_bins):
        bin_mask = bin_indices == i
        
        if np.sum(bin_mask) > 0:
            ys_in_bin = ys[bin_mask]
            xs_in_bin = xs[bin_mask]
            
            threshold = np.percentile(ys_in_bin, pct)
            below_threshold = ys_in_bin <= threshold
            
            npil_est_xs.extend(xs_in_bin[below_threshold])
            npil_est_ys.extend(ys_in_bin[below_threshold])
    
    npil_est_xs = np.array(npil_est_xs)
    npil_est_ys = np.array(npil_est_ys)
    
    # Perform linear regression to estimate coefficient and intercept
    if len(npil_est_xs) > 1:
        coefficient, intercept = np.polyfit(npil_est_xs, npil_est_ys, 1)
    else:
        coefficient = 0.0
        intercept = 0.0

    if return_points:
        return coefficient, intercept, npil_est_xs, npil_est_ys
    else:
        return coefficient, intercept

# now write a "smart" version that does this for all cells WITHOUT using a loop! in the function above input F is of size n_t but in this new function input F is of size n_cells x n_t
def est_npilcoeff_pctregression_all(F, Fneu, pct=5, neu_bins=20, return_mask=False, eps=1e-12):
    """
    Vectorized percentile-regression neuropil coefficient estimation for all cells.

    For each cell independently, perform a linear regression y = a x + b
    between fluorescence y = F[c] and neuropil x = Fneu[c], using only the
    lowest `pct` percentile of y-values within equal-width bins of x.

    Bins are per-cell: we normalize x to [0, 1] using each cell's min/max and
    apply common edges there (equivalent to per-cell min/max binning).

    Parameters
    ---------
    F : ndarray, shape (n_cells, n_t)
        Fluorescence traces per cell.
    Fneu : ndarray, shape (n_cells, n_t)
        Neuropil traces per cell.
    pct : float, default 5
        Percentile threshold within each bin (0..100).
    neu_bins : int, default 20
        Number of equal-width neuropil bins per cell.
    return_mask : bool, default False
        When True, also return the boolean mask (n_cells, n_t) of selected
        timepoints used for the regression in each cell.
    eps : float, default 1e-12
        Numerical stabilizer for per-cell normalization (avoids divide-by-zero).

    Returns
    -------
    coefficients : ndarray, shape (n_cells,)
        Estimated neuropil coefficients (slopes) per cell.
    intercepts : ndarray, shape (n_cells,)
        Estimated intercepts per cell.
    mask : ndarray, shape (n_cells, n_t), optional
        Only returned when return_mask=True. Selected timepoints per cell.

    Notes
    -----
    - Avoids Python loops entirely by using NumPy vectorization.
    - If a cell has <2 selected points or degenerate x-variance, (a, b) = (0, 0).
    """

    F = np.asarray(F)
    Fneu = np.asarray(Fneu)
    if F.ndim != 2 or Fneu.ndim != 2:
        raise ValueError("F and Fneu must be 2D arrays of shape (n_cells, n_t)")
    if F.shape != Fneu.shape:
        raise ValueError("F and Fneu must have the same shape")

    n_cells, n_t = F.shape

    # Per-cell normalization of neuropil to [0, 1] to replicate per-cell binning
    x_min = Fneu.min(axis=1, keepdims=True)
    x_ptp = np.ptp(Fneu,axis=1, keepdims=True)
    x_norm = (Fneu - x_min) / (x_ptp + eps)

    # Assign bins using common edges on [0, 1]
    edges = np.linspace(0.0, 1.0, neu_bins + 1)
    bin_idx = np.searchsorted(edges, x_norm.ravel(), side='right') - 1
    bin_idx = np.clip(bin_idx, 0, neu_bins - 1).reshape(n_cells, n_t)

    # Rank y within each (cell, bin) group (no loops):
    # Sort per cell by (bin_idx asc, F asc)
    order = np.lexsort((F, bin_idx))  # shape: (n_cells, n_t)
    sorted_bins = np.take_along_axis(bin_idx, order, axis=1)

    # Start-of-group flags in sorted order
    first_col = sorted_bins[:, :1]
    group_start_flag = np.concatenate(
        [np.ones_like(first_col, dtype=bool), sorted_bins[:, 1:] != sorted_bins[:, :-1]],
        axis=1,
    )
    pos = np.arange(n_t, dtype=np.int32)[None, :]
    group_start_pos = np.maximum.accumulate(np.where(group_start_flag, pos, -1), axis=1)
    rank_sorted = pos - group_start_pos

    # Invert permutation to map ranks back to original time order
    inv_order = np.empty_like(order)
    row_idx = np.arange(n_cells)[:, None]
    inv_order[row_idx, order] = pos
    rank = np.take_along_axis(rank_sorted, inv_order, axis=1)

    # Counts per (cell, bin) via one-hot accumulation
    counts = (bin_idx[..., None] == np.arange(neu_bins)[None, None, :]).sum(axis=1)
    r_thresh_bin = np.floor((pct / 100.0) * (counts - 1)).astype(np.int32)
    r_thresh_bin = np.maximum(r_thresh_bin, -1)
    r_thresh = r_thresh_bin[row_idx, bin_idx]

    # Selected points: lowest pct percentile of y within each bin
    selected = rank <= r_thresh

    # OLS on selected subset per cell
    m = selected.astype(np.float64)
    S1 = m.sum(axis=1)
    Sx = (Fneu * m).sum(axis=1)
    Sy = (F * m).sum(axis=1)
    Sxx = (Fneu * Fneu * m).sum(axis=1)
    Sxy = (Fneu * F * m).sum(axis=1)

    den = S1 * Sxx - Sx * Sx
    valid = (S1 >= 2) & (den != 0)

    coefficients = np.zeros(n_cells, dtype=np.float64)
    intercepts = np.zeros(n_cells, dtype=np.float64)
    coefficients[valid] = (S1[valid] * Sxy[valid] - Sx[valid] * Sy[valid]) / den[valid]
    intercepts[valid] = (Sy[valid] - coefficients[valid] * Sx[valid]) / S1[valid]

    if return_mask:
        return coefficients, intercepts, selected
    return coefficients, intercepts