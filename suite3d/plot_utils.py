import numpy as n
import os
import matplotlib as mpl
from scipy import stats
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d,uniform_filter1d
from .developer import todo


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl



def diverging_cmap(low_color="blue", high_color="red", mid_color="white", name="diverging", nan_color="lightgrey"):
    """
    Creates a diverging colormap with a specified color at the center (zero value)
    and a color for NaN values.

    Args:
        low_color (str): Color for the low end of the colormap.
        high_color (str): Color for the high end of the colormap.
        mid_color (str): Color for the center (zero) of the colormap.
        name (str): Name of the colormap.
        nan_color (str): Color to display for NaN values.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The created colormap.
    """
    if mid_color is None:
        cdict = {
            "red": [
                (0.0, mcolors.to_rgb(low_color)[0], mcolors.to_rgb(low_color)[0]),
                (1.0, mcolors.to_rgb(high_color)[0], mcolors.to_rgb(high_color)[0]),
            ],
            "green": [
                (0.0, mcolors.to_rgb(low_color)[1], mcolors.to_rgb(low_color)[1]),
                (1.0, mcolors.to_rgb(high_color)[1], mcolors.to_rgb(high_color)[1]),
            ],
            "blue": [
                (0.0, mcolors.to_rgb(low_color)[2], mcolors.to_rgb(low_color)[2]),
                (1.0, mcolors.to_rgb(high_color)[2], mcolors.to_rgb(high_color)[2]),
            ],
        }
    else:
        cdict = {
            "red": [
                (0.0, mcolors.to_rgb(low_color)[0], mcolors.to_rgb(low_color)[0]),
                (0.5, mcolors.to_rgb(mid_color)[0], mcolors.to_rgb(mid_color)[0]),
                (1.0, mcolors.to_rgb(high_color)[0], mcolors.to_rgb(high_color)[0]),
            ],
            "green": [
                (0.0, mcolors.to_rgb(low_color)[1], mcolors.to_rgb(low_color)[1]),
                (0.5, mcolors.to_rgb(mid_color)[1], mcolors.to_rgb(mid_color)[1]),
                (1.0, mcolors.to_rgb(high_color)[1], mcolors.to_rgb(high_color)[1]),
            ],
            "blue": [
                (0.0, mcolors.to_rgb(low_color)[2], mcolors.to_rgb(low_color)[2]),
                (0.5, mcolors.to_rgb(mid_color)[2], mcolors.to_rgb(mid_color)[2]),
                (1.0, mcolors.to_rgb(high_color)[2], mcolors.to_rgb(high_color)[2]),
            ],
        }
    cmap = mcolors.LinearSegmentedColormap(name, cdict)
    cmap.set_bad(nan_color)
    return cmap


def multiple_timeseries(
    ts,
    yss,
    colors=None,
    labels=None,
    alphas=None,
    lws=None,
    do_zscore=True,
    dy=3.0,
    auto_ylim=True,
    ax=None,
    tick_labels=True,
    legend=False,
    lw=1.0,
    alpha=1.0,
    color=None,
    do_filt=None,
    swap_yorder=False,
    ylabel_rot=0,
    dy_offset=0,
    yposs=None,
    idx_lims=None,
    tlims=None,
):
    if ax is None:
        f, ax = plt.subplots()
    n_lines = len(yss)
    yticks = []
    lines = []

    if len(n.shape(ts)) == 0:
        ts = n.arange(len(yss[0])) * ts

    if tlims is not None:
        idx0 = n.argmin(n.abs(ts - tlims[0]))
        idx1 = n.argmin(n.abs(ts - tlims[1]))
        idx_lims = (idx0, idx1)

    if idx_lims is not None:
        ts = ts[idx_lims[0] : idx_lims[1]]
    for i in range(n_lines):
        color = colors[i] if colors is not None else color
        alpha = alphas[i] if alphas is not None else alpha
        label = labels[i] if labels is not None else None
        lw = lws[i] if lws is not None else lw
        ys = yss[i]

        if idx_lims is not None:
            ys = ys[idx_lims[0] : idx_lims[1]]

        if do_zscore:
            ys = zscore(ys)
        if do_filt is not None:
            ys = filt(ys, do_filt)
        if yposs is None:
            ypos = dy * i + dy_offset
            if swap_yorder:
                ypos = dy * (n_lines - 1 - i)
        else:
            ypos = yposs[i]
        lines += ax.plot(
            ts, ys + ypos, color=color, alpha=alpha, linewidth=lw, label=label
        )
        yticks.append(ypos)
    if labels is not None and legend:
        ax.legend(lines[::-1], labels[::-1], frameon=True, facecolor="white")
    ax.set_yticks(yticks)
    if labels is not None and tick_labels:
        ax.set_yticklabels(labels, rotation=ylabel_rot, va="center")
    else:
        ax.set_yticklabels([""] * len(yticks))

    if auto_ylim:
        ax.set_ylim(-dy, dy * (i + 1))

    ax.set_xlim(ts.min(), ts.max())
    return ax


def plot_onsets(onset_times, offset_times, ax, alpha=0.5, color="grey"):
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    for i in range(len(onset_times)):
        patch1 = ax.fill_between(
            [onset_times[i], offset_times[i]], *ylim, color=color, alpha=alpha
        )
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    return patch1


def zscore(
    x, nax=0, m=None, std=None, return_params=False, auto_reshape=True, undo=False
):
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
        m = n.nanmean(x,axis=axes_to_reduce, keepdims=True)
    if std is None:
        std = n.nanstd(x,axis=axes_to_reduce, keepdims=True)
    print(m,std)
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


def filt(signal, width=3, axis=0, mode="gaussian"):
    """
    apply a simple filter to a 1d signal

    Args:
        signal (ndarray): ndim ndarray
        width (int, optional): Width of filter. Defaults to 3.
        axis (int, optional): axis to apply filter on. Defaults to 0.
        mode (str, optional): Type of filter. 'gaussian' or 'uniform'

    Returns:
        signal: same shape as input, filtered
    """
    if width == 0:
        return signal

    if mode == "gaussian":
        out = gaussian_filter1d(signal, sigma=width, axis=axis)
    elif mode == "uniform":
        out = uniform_filter1d(signal, size=width, axis=axis)
    else:
        assert False, "mode not implemented"
    return out


def hist2d(
    xs,
    ys,
    nbins=51,
    xlims=None,
    ylims=None,
    regression=True,
    ax=None,
    log=True,
    cbar=True,
    cmap="Blues",
    density=False,
    clims=(None, None),
    plot_identity=False,
    slope_in_label=True,
    xbins=None,
    ybins=None,
    regression_line_params={},
    fix_nans=True,
    lim_percentile=None,
):
    if fix_nans:
        nans = n.isnan(xs) | n.isnan(ys)
        if nans.sum() > 0:
            xs = xs[~nans]
            ys = ys[~nans]
    if ax is None:
        f, ax = plt.subplots(figsize=(6, 6))

    if xlims is None:
        xlims = xs.min(), xs.max()
    if ylims is None:
        ylims = ys.min(), ys.max()

    if xbins is None:
        if lim_percentile:
            xbins = n.linspace(
                n.percentile(xs, lim_percentile),
                n.percentile(xs, 100 - lim_percentile),
                nbins,
            )
        else:
            xbins = n.linspace(*xlims, nbins)
    if ybins is None:
        if lim_percentile:
            ybins = n.linspace(
                n.percentile(ys, lim_percentile),
                n.percentile(ys, 100 - lim_percentile),
                nbins,
            )
        else:
            ybins = n.linspace(*ylims, nbins)

    if log:
        norm = mpl.colors.LogNorm(vmin=clims[0], vmax=clims[1])
    else:
        norm = mpl.colors.Normalize(vmin=clims[0], vmax=clims[1])

    if clims is None:
        clims = (None, None)

    hist = ax.hist2d(xs, ys, bins=(xbins, ybins), cmap=cmap, norm=norm, density=density)

    if cbar:
        plt.colorbar(hist[-1], ax=ax)

    if plot_identity:
        ax.plot(xlims, xlims, color="k", alpha=0.2, lw=3, linestyle="--")

    if regression:
        slopex, interceptx, rx, px, __ = stats.linregress(xs, ys)
        if slope_in_label:
            label = label = "y=%.2fx + %.2f\nR (CoD) : %.2f" % (slopex, interceptx, rx)
        else:
            label = "R: %.2f" % rx
        ax.plot(
            xbins,
            xbins * slopex + interceptx,
            color="k",
            label=label,
            **regression_line_params,
        )
        ax.legend()
    # print(hist[-1].get_clim())

    return ax


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


def show_vol_all_planes(
    img,
    figsize=(8, 6),
    title=None,
    suptitle=None,
    ncols=5,
    same_scale=False,
    vminmax_percentile=(0.5, 99.5),
    vminmax=None,
    **kwargs,
):
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

    figsize = figsize  # ideally multiple of rows and colloumns
    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize), layout="constrained")
    if ncols == 1 or nrows == 1:
        if nrows == 1 and ncols == 1:
            axs = [[axs]]
        else:
            axs = [axs]
    # make all the images have the same color scale
    if same_scale:
        if vminmax is None:
            todo("whats going on here?")
            for z in range(nz):
                if z == 0:
                    non_nan = ~n.isnan(img[z])
                    vmin = n.percentile(img[z][non_nan], vminmax_percentile[0])
                    vmax = n.percentile(img[z][non_nan], vminmax_percentile[1])
                else:
                    non_nan = ~n.isnan(img[z])
                    vmin = min(vmin, n.percentile(img[z][non_nan], vminmax_percentile[0]))
                    vmax = max(vmax, n.percentile(img[z][non_nan], vminmax_percentile[1]))
            vminmax = (vmin, vmax)
    for row in range(nrows):
        for col in range(ncols):
            plane_no = row * ncols + col
            if plane_no < nz:  # catch empty planes
                if same_scale:
                    show_img(img[plane_no], ax=axs[row][col], vminmax=vminmax, **kwargs)
                else:
                    show_img(
                        img[plane_no],
                        ax=axs[row][col],
                        vminmax_percentile=vminmax_percentile,
                        vminmax=vminmax,
                        **kwargs,
                    )
                if title is None:
                    axs[row][col].set_title(
                        f"Plane {plane_no + 1}", fontsize="small"
                    )  # Counting from 0
                else:
                    axs[row][col].set_title(title[plane_no])
            else:
                # hide axis for empty planes
                axs[row][col].set_axis_off()

    if suptitle is not None:
        fig.suptitle(suptitle)
    return fig, axs


def show_img(
    im,
    flip=1,
    cmap="Greys_r",
    colorbar=False,
    other_args={},
    figsize=(8, 6),
    dpi=150,
    alpha=None,
    return_fig=True,
    ticks=False,
    ax=None,
    px_py=None,
    exact_pixels=False,
    vminmax_percentile=(0.5, 99.5),
    vminmax=None,
    facecolor="white",
    xticks=None,
    yticks=None,
    norm=None,
    cbar=False,
    cbar_loc="left",
    cbar_fontcolor="k",
    cbar_ori="vertical",
    cbar_title="",
    interpolation="nearest",
    ax_off=False,
    cax_kwargs={"frameon": False},
):

    f = None
    im = im.copy()
    if exact_pixels:
        ny, nx = im.shape
        figsize = (nx / dpi, ny / dpi)
        px_py = None

    new_args = {}
    new_args.update(other_args)
    if ax is None:
        f, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if facecolor is not None:
        ax.set_facecolor(facecolor)
    ax.grid(False)
    new_args["interpolation"] = interpolation
    if vminmax_percentile is not None and vminmax is None:
        non_nan = ~n.isnan(im)
        new_args["vmin"] = n.percentile(im[non_nan], vminmax_percentile[0])
        new_args["vmax"] = n.percentile(im[non_nan], vminmax_percentile[1])
    if vminmax is not None:
        new_args["vmin"] = vminmax[0]
        new_args["vmax"] = vminmax[1]
    if px_py is not None:
        new_args["aspect"] = px_py[1] / px_py[0]
    if alpha is not None:
        if type(alpha) == float:
            alpha = n.ones_like(im) * alpha
        new_args["alpha"] = alpha.copy()
    if norm is not None:
        new_args["norm"] = norm
        new_args["vmin"] = None
        new_args["vmax"] = None
    # print(new_args)
    axim = ax.imshow(flip * im, cmap=cmap, **new_args)
    if colorbar:
        plt.colorbar()
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if exact_pixels:
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # plt.tight_layout()
    if norm:
        new_args["vmin"] = norm.vmin
        new_args["vmax"] = norm.vmax
    if cbar:
        if cbar_loc == "left":
            cbar_loc = [0.025, 0.4, 0.02, 0.2]
            cbar_ori = "vertical"
        elif cbar_loc == "right":
            cbar_loc = [0.88, 0.4, 0.02, 0.2]
            cbar_ori = "vertical"
        elif cbar_loc == "top":
            cbar_loc = [0.4, 0.95, 0.2, 0.02]
            cbar_ori = "horizontal"
        elif cbar_loc == "bottom":
            cbar_loc = [0.4, 0.05, 0.2, 0.02]
            cbar_ori = "horizontal"
        cax = ax.inset_axes(cbar_loc, **cax_kwargs)
        plt.colorbar(axim, cax=cax, orientation=cbar_ori)
        if cbar_ori == "vertical":
            cax.set_yticks(
                [new_args["vmin"], new_args["vmax"]],
                ["%.2f" % new_args["vmin"], "%.2f" % new_args["vmax"]],
                color=cbar_fontcolor,
                fontsize=9,
            )
            cax.set_ylabel(cbar_title, color=cbar_fontcolor, fontsize=9, labelpad=-13)
        if cbar_ori == "horizontal":
            cax.set_xticks(
                [new_args["vmin"], new_args["vmax"]],
                ["%.2f" % new_args["vmin"], "%.2f" % new_args["vmax"]],
                color=cbar_fontcolor,
                fontsize=9,
            )
            cax.set_xlabel(cbar_title, color=cbar_fontcolor, fontsize=9, labelpad=-13)
    if xticks is not None:
        ax.set_xticks(range(len(xticks)), xticks)
    if yticks is not None:
        ax.set_yticks(range(len(yticks)), yticks)
    if ax_off:
        ax.axis("off")

    if return_fig:
        return f, ax, axim



def turn_off_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)



from IPython.display import display
import ipywidgets as ipyw


# https://github.com/mohakpatel/ImageSliceViewer3D/blob/master/ImageSliceViewer3D.ipynb
class VolumeViewer:
    """
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks.

    User can interactively change the slice plane selection for the image and
    the slice plane being viewed.

    Argumentss:
    Volume = 3D input image
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html

    """

    def __init__(
        self,
        volume,
        figsize=(8, 8),
        cmap="Greys_r",
        vminmax=None,
        overlay=None,
        z0=None,
        **kwargs,
    ):
        self.volume = volume
        self.overlay = overlay
        self.figsize = figsize
        self.cmap = cmap
        self.kwargs = kwargs

        if vminmax is None:
            self.v = n.percentile(volume.flatten(), 0.5), n.percentile(
                volume.flatten(), 99.5
            )
        else:
            self.v = vminmax

        # self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)

        # Call to select slice plane
        ipyw.interact(
            self.view_selection,
            view=ipyw.RadioButtons(
                options=["x-y", "y-z", "z-x"],
                value="x-y",
                description="Slice plane selection:",
                disabled=False,
                style={"description_width": "initial"},
            ),
        )

    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z": [1, 2, 0], "z-x": [2, 0, 1], "x-y": [0, 1, 2]}
        self.vol = n.transpose(self.volume, orient[view])
        if self.overlay is not None:
            self.overlay_vol = n.transpose(self.overlay, orient[view] + [3])
        print(self.vol.shape)
        maxZ = self.vol.shape[0] - 1

        # Call to view a slice within the selected slice plane
        ipyw.interact(
            self.plot_slice,
            z=ipyw.IntSlider(
                min=0,
                max=maxZ,
                value=int(maxZ // 2),
                step=1,
                continuous_update=False,
                description="Image Slice:",
            ),
            show_overlay=True,
        )

    def plot_slice(self, z, show_overlay=True):
        display(f"UPDATING to z: {z}")
        # Plot slice for the given plane and slice
        __, ax, __ = show_img(
            self.vol[z, :, :],
            cmap=self.cmap,
            vminmax=self.v,
            figsize=self.figsize,
            **self.kwargs,
        )

        if show_overlay and self.overlay is not None:
            show_img(self.overlay_vol[z], ax=ax)
        # plt.show()
        # self.fig = plt.figure(figsize=self.figsize)
        # plt.imshow(
        #     self.vol[:, :, z],
        #     cmap=plt.get_cmap(self.cmap),
        #     vmin=self.v[0],
        #     vmax=self.v[1],
        # )


# existing code...
def density_scatter(
    x, y,
    *,
    cmap: str = "viridis",
    ax=None,
    s: float = 10,
    cbar: bool = False,
    density: str = 'hist',
    density_bins: int = 64,      # for 'hist'
    gaussian_sigma: float | int = 0,  # smoothing on histogram (pixels)
    knn_k: int = 20,              # for 'knn'
    log_scale: bool = False,      # use logarithmic color scale
    # colorbar inside-axis options
    cbar_loc: str = 'lower right',
    cbar_size: str = '3%',
    cbar_height: str = '20%',
    cbar_borderpad: float = 0.2,
    cbar_orientation: str = 'vertical',
    # identity line options
    identity_line: bool = False,
    max_pts = 5000, # max points to plot, randomly subsampled if more
    # statistics / legend options
    show_stats: bool = False,
    stats_loc: str = 'best',
    stats_fmt: str = 'slope={slope:.3g}, r={r:.3g}, p={p:.1e}',
    stats_frameon: bool = False,
    **scatter_kwargs
):
    """Scatter plot colored by local point density.

    Parameters
    ----------
    x, y : array-like
        1D arrays of the same length.
    density : {'gaussian','hist','knn','uniform'}
        - 'gaussian': scipy.stats.gaussian_kde (slow for large n)
        - 'hist': 2D histogram (+ optional Gaussian blur) then per-point lookup
        - 'knn': k-NN density via cKDTree using 1/(pi r_k^2)
        - 'uniform': constant color (fallback)
    density_bins : int
        Number of bins per axis for 'hist'.
    gaussian_sigma : float
        Gaussian blur sigma (in bins) for 'hist'. 0 disables smoothing.
    knn_k : int
        k for k-NN density.
    log_scale : bool
        If True, use a logarithmic color scale (matplotlib.colors.LogNorm).
    identity_line : bool
        If True, draw a gray dashed identity line (y = x) behind the scatter without
        changing axis limits.
    show_stats : bool
        If True, compute linear regression (scipy.stats.linregress) and add a legend entry
        containing slope, Pearson r, and p-value using stats_fmt.
    stats_loc : str
        Matplotlib legend location for the stats string (if show_stats=True).
    stats_fmt : str
        Format string with placeholders {slope}, {r}, {p}, {intercept}.
    stats_frameon : bool
        Whether the legend frame is shown when displaying stats.
    """
    # Convert and clean inputs
    x = n.asarray(x).ravel()
    y = n.asarray(y).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    
    if x.size > max_pts:
        idx = n.random.choice(x.size, size=max_pts, replace=False)
        x = x[idx]
        y = y[idx]

    mask = n.isfinite(x) & n.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        raise ValueError("No finite points to plot")

    # Compute point density
    xy = n.vstack([x, y])
    z = None
    if density == 'gaussian':
        try:
            kde = stats.gaussian_kde(xy)
            z = kde(xy)
        except Exception:
            # Fallback to fast histogram method if KDE fails (e.g., singular covariance)
            density = 'hist'

    if density == 'hist':
        # 2D histogram on a grid
        H, xedges, yedges = n.histogram2d(x, y, bins=density_bins)
        if gaussian_sigma and gaussian_sigma > 0:
            from scipy.ndimage import gaussian_filter
            H = gaussian_filter(H, gaussian_sigma, mode='constant')
        # Map each point to its bin count (fast)
        ix = n.clip(n.digitize(x, xedges) - 1, 0, H.shape[0] - 1)
        iy = n.clip(n.digitize(y, yedges) - 1, 0, H.shape[1] - 1)
        z = H[ix, iy] + 1e-12  # avoid zeros

    elif density == 'knn':
        # k-NN density estimate using area of circle to k-th neighbor
        from scipy.spatial import cKDTree
        tree = cKDTree(n.c_[x, y])
        dists, _ = tree.query(n.c_[x, y], k=knn_k + 1)  # include self
        rk = dists[:, -1]
        area = n.pi * n.maximum(rk, 1e-12) ** 2
        z = 1.0 / area

    elif density == 'uniform':
        z = n.full_like(x, fill_value=1.0 / max(1, x.size), dtype=float)

    if z is None:
        raise ValueError("Unknown density method. Use 'gaussian', 'hist', 'knn', or 'uniform'.")

    # Sort so densest points are plotted last
    idx = n.argsort(z)
    x_sorted = x[idx]
    y_sorted = y[idx]
    z_sorted = z[idx]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
        created_fig = True
    else:
        fig = None

    # Apply logarithmic normalization if requested (unless user already provided a norm)
    if log_scale and ('norm' not in scatter_kwargs):
        # Ensure strictly positive vmin for LogNorm
        zpos = z_sorted[z_sorted > 0]
        if zpos.size == 0:
            zpos = n.array([1.0])
        vmin = scatter_kwargs.get('vmin', float(zpos.min()))
        vmax = scatter_kwargs.get('vmax', float(z_sorted.max()))
        scatter_kwargs['norm'] = mpl.colors.LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, vmin * 1.000001))

    sc = ax.scatter(x_sorted, y_sorted, c=z_sorted, s=s, cmap=cmap, **scatter_kwargs)

    # Optional identity line (y=x) behind points; restore limits so it doesn't affect view
    if identity_line:
        xlim0 = ax.get_xlim()
        ylim0 = ax.get_ylim()
        lo = float(min(xlim0[0], ylim0[0]))
        hi = float(max(xlim0[1], ylim0[1]))
        try:
            zbase = float(sc.get_zorder())
        except Exception:
            zbase = 1.0
        ax.plot([lo, hi], [lo, hi], color='0.6', linestyle='--', linewidth=1.0, zorder=zbase - 1)
        ax.set_xlim(xlim0)
        ax.set_ylim(ylim0)

    # Optional stats legend (after plotting so limits unaffected)
    if show_stats and x.size > 1:
        try:
            lr = stats.linregress(x, y)
            label = stats_fmt.format(slope=lr.slope, r=lr.rvalue, p=lr.pvalue, intercept=lr.intercept)
            sc.set_label(label)
            # Only draw legend if not already present (or user wants it explicitly)
            existing_legend = ax.get_legend()
            if existing_legend is None:
                ax.legend(loc=stats_loc, frameon=stats_frameon)
        except Exception:
            # Silently ignore regression errors (e.g., constant input)
            pass
    if cbar:
        # Create an inset colorbar inside the plotting axes using fixed bounds to avoid
        # AnchoredLocator issues during save/render.
        fig_for_cb = ax.figure if ax is not None else plt.gcf()

        def _as_frac(v, default_frac):
            # Convert values like '3%' -> 0.03, numbers <=1 kept as-is, >1 treated as percent.
            if isinstance(v, str) and v.endswith('%'):
                try:
                    return float(v[:-1]) / 100.0
                except Exception:
                    return default_frac
            try:
                vf = float(v)
                if vf <= 1.0:
                    return vf
                # Treat e.g. 3 as 3%
                return vf / 100.0
            except Exception:
                return default_frac

        w_frac = _as_frac(cbar_size, 0.03)
        h_frac = _as_frac(cbar_height, 0.4)
        pad = float(cbar_borderpad) if cbar_borderpad is not None else 0.02
        # If pad looks like inches (large), clamp to a small fraction
        if pad > 0.5:
            pad = 0.02

        # Compute bounds in axes fraction coordinates based on location keyword
        loc = (cbar_loc or 'upper right').lower()
        if loc == 'upper right':
            x0 = 1 - w_frac - pad
            y0 = 1 - h_frac - pad
        elif loc == 'upper left':
            x0 = pad
            y0 = 1 - h_frac - pad
        elif loc == 'lower right':
            x0 = 1 - w_frac - pad
            y0 = pad
        elif loc == 'lower left':
            x0 = pad
            y0 = pad
        else:
            # fallback: upper right
            x0 = 1 - w_frac - pad
            y0 = 1 - h_frac - pad

        cbax = ax.inset_axes([x0, y0, w_frac, h_frac])
        cbar_obj = mpl.colorbar.Colorbar(cbax, sc, orientation=cbar_orientation)
        # Optional: keep the colorbar tidy inside the axis
        for spine in cbax.spines.values():
            spine.set_linewidth(0.5)
    return fig, ax, sc
# ...existing code...

