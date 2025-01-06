import numpy as n
import os
import matplotlib as mpl
from scipy import stats
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d


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


def show_tif_all_planes(
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
                    show_tif(img[plane_no], ax=axs[row][col], vminmax=vminmax, **kwargs)
                else:
                    show_tif(
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


def show_tif(
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
                ["%.3f" % new_args["vmin"], "%.3f" % new_args["vmax"]],
                color=cbar_fontcolor,
                fontsize=9,
            )
            cax.set_ylabel(cbar_title, color=cbar_fontcolor, fontsize=9, labelpad=-13)
        if cbar_ori == "horizontal":
            cax.set_xticks(
                [new_args["vmin"], new_args["vmax"]],
                ["%.3f" % new_args["vmin"], "%.3f" % new_args["vmax"]],
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
    transpose=False,
    bin=None,
    ticks=False,
    ax=None,
    px_py=None,
    exact_pixels=False,
    vminmax_percentile=(0.5, 99.5),
    vminmax=None,
    facecolor="white",
    xticks=None,
    yticks=None,
    return_cax=False,
    cbar_fontsize=9,
    norm=None,
    cbar=False,
    cbar_loc="left",
    cbar_ori="vertical",
    cbar_title="",
    interpolation="nearest",
    ax_off=False,
    cax_kwargs={"frameon": False},
    extent=None,
    spines=False,
    symmetric_cmap=False,
    aspect=None,
    cax_fontcolor="k",
    cax_label_format="%.2f",
):

    f = None
    im = im.copy()
    if type(bin) is tuple:
        im = math.bin1d(math.bin1d(im, bin[0], axis=0), bin[1], axis=1)
    elif bin is not None:
        im = math.bin1d(math.bin1d(im, bin, axis=0), bin, axis=1)
        # print("binned")
    if transpose:
        # print(im.shape)
        im = n.moveaxis(im, (0, 1), (1, 0))
        # print(im.shape)
        if alpha is not None:
            alpha = alpha.T
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
        new_args["vmin"] = n.nanpercentile(im, vminmax_percentile[0])
        new_args["vmax"] = n.nanpercentile(im, vminmax_percentile[1])
    if vminmax is not None:
        new_args["vmin"] = vminmax[0]
        new_args["vmax"] = vminmax[1]
    if symmetric_cmap:
        vmax_abs = max(n.abs(new_args["vmin"]), n.abs(new_args["vmax"]))
        new_args["vmin"] = -vmax_abs
        new_args["vmax"] = vmax_abs
    if px_py is not None:
        new_args["aspect"] = px_py[1] / px_py[0]
    if aspect is not None:
        new_args["aspect"] = aspect
    if alpha is not None:
        new_args["alpha"] = alpha.copy()
    if norm is not None:
        new_args["norm"] = norm
        new_args["vmin"] = None
        new_args["vmax"] = None
    if extent is not None:
        new_args["extent"] = extent
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
        if cbar_loc == "lower left":
            cbar_loc = [0.025, 0.025, 0.02, 0.2]
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
                [
                    cax_label_format % new_args["vmin"],
                    cax_label_format % new_args["vmax"],
                ],
                color=cax_fontcolor,
                fontsize=cbar_fontsize,
            )
            cax.set_ylabel(
                cbar_title, color=cax_fontcolor, fontsize=cbar_fontsize, labelpad=-10
            )
        if cbar_ori == "horizontal":
            cax.set_xticks(
                [new_args["vmin"], new_args["vmax"]],
                [
                    cax_label_format % new_args["vmin"],
                    cax_label_format % new_args["vmax"],
                ],
                color=cax_fontcolor,
                fontsize=cbar_fontsize,
            )
            cax.set_xlabel(
                cbar_title, color=cax_fontcolor, fontsize=cbar_fontsize, labelpad=-10
            )
    if xticks is not None:
        ax.set_xticks(range(len(xticks)), xticks)
    if yticks is not None:
        ax.set_yticks(range(len(yticks)), yticks)
    if ax_off:
        ax.axis("off")
    if not spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    if return_cax:
        return f, ax, axim, cax
    if return_fig:
        return f, ax, axim


def turn_off_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
