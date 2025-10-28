import os
import numpy as n
from multiprocessing import Pool
from scipy.signal import find_peaks
from scipy.stats import gamma
import matplotlib.pyplot as plt

# from matplotlib import colors
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    import imreg_dft as imreg
except:
    print("No Imreg DFT")
from multiprocessing import Pool, shared_memory
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import convolve1d

from itertools import product

try:
    from dask import array as darr
    from dask_image import ndfilters as dafilt
    from skimage.measure import moments
    from skimage.metrics import normalized_mutual_information
except:
    print("Missing some packages")
from datetime import datetime
from multiprocessing import cpu_count
import time

from .developer import deprecated

# Only show this error once
try:
    from git import Repo
except ImportError:
    pass

colors = ["#90be6d", "#e98a15", "#b26c98", "#1b9aaa", "#3a405a"]


def set_num_processors(n_procs):
    """
    A function to set the number of processors to use for parallel processing. The argument
    is the requested number of processors, and this function makes sure that the number is
    greater than or equal to 1 and less than or equal to the number of processors in the
    computer minus 1 (as reported by multiprocessing.cpu_count()).

    Args:
        n_procs (int): The number of processors to use for parallel processing.
    """
    return max(min(n_procs, cpu_count() - 1), 1)


def spatial_downsample_movie(mov, factor):
    # mov of size (nz, nt, ny, nx)
    # factor is an integer
    dafilt.uniform_filter(mov, size=(1, 1, factor, factor), mode="nearest", cval=0.0)
    mov = mov[:, :, ::factor, ::factor]
    return mov


def pad_and_fuse(mov, plane_shifts, fuse_shift, xs, fuse_shift_offset=0):
    nz, nt, nyo, nxo = mov.shape
    n_stitches = len(xs) - 1
    n_xpix_lost_for_fusing = n_stitches * fuse_shift

    plane_shifts = n.round(plane_shifts).astype(int)

    xrange = plane_shifts[:, 1].min(), plane_shifts[:, 1].max()
    yrange = plane_shifts[:, 0].min(), plane_shifts[:, 0].max()

    ypad = n.ceil(n.abs(n.diff(yrange))).astype(int)[::-1]
    yshift = n.ceil(n.abs((yrange[0]))).astype(int)
    xpad = n.ceil(n.abs(n.diff(xrange))).astype(int)[::-1]
    xshift = n.ceil(n.abs((xrange[0]))).astype(int)
    nyn = nyo + ypad.sum()
    nxn = nxo + xpad.sum() - n_xpix_lost_for_fusing

    mov_pad = n.zeros((nz, nt, nyn, nxn), n.float32)

    lshift = fuse_shift // 2 - fuse_shift_offset
    rshift = fuse_shift - lshift
    xn0 = 0
    og_xs = []
    new_xs = []
    for i in range(n_stitches + 1):
        x0 = xs[i]
        if i > 0:
            x0 += lshift
        if i == n_stitches:
            x1 = nxo
        else:
            x1 = xs[i + 1] - rshift
        dx = x1 - x0
        # print(x0,x1, xn0, xn0+dx, mov_pad.shape, mov.shape)
        mov_pad[:, :, yshift : yshift + nyo, xshift + xn0 : xshift + xn0 + dx] = mov[:, :, :, x0:x1]
        new_xs.append((xn0, xn0 + dx))
        og_xs.append((x0, x1))
        xn0 += dx
    return mov_pad, xpad, ypad, new_xs, og_xs


def edge_crop_movie(mov, summary=None, edge_crop_npix=None):
    """
    Set the edges of each plane to 0 to prevent artifacts due to registration

    Args:
        mov (ndarray): nt, nz, ny, nx - watch out for the shape!
        summary (dict, optional): output of job.load_summary(). Defaults to None.
        edge_crop_npix (int, optional): number of pixels to set to 0 on each edge. Defaults to None.

    Returns:
        mov: ndarray edge-cropped movie (operation done inplace)
    """
    if edge_crop_npix is None or edge_crop_npix < 1:
        return mov
    __, nz, ny, nx = mov.shape
    yt, yb, xl, xr = get_shifted_plane_bounds(summary["plane_shifts"], ny, nx, summary["ypad"][0], summary["xpad"][0])
    for i in range(nz):
        mov[:, i, : yt[i] + edge_crop_npix] = 0
        mov[:, i, yb[i] - edge_crop_npix :] = 0
        mov[:, i, :, : xl[i] + edge_crop_npix] = 0
        mov[:, i, :, xr[i] - edge_crop_npix :] = 0

    return mov


def get_shifted_plane_bounds(plane_shifts, ny, nx, ypad, xpad):
    """return the top/bottom and left/right borders of each plane
    that has been shifted by pad_and_fuse"""
    y_bottoms = []
    y_tops = []
    x_lefts = []
    x_rights = []

    ny_og = ny - ypad
    nx_og = nx - xpad
    ydir, xdir = n.sign(plane_shifts.mean(axis=0))
    for plane_idx in range(plane_shifts.shape[0]):
        y_shift, x_shift = n.round(plane_shifts[plane_idx]).astype(int)
        if xdir < 0:
            x_left = xpad + x_shift
            x_right = nx + (x_shift)
        else:
            x_left = x_shift
            x_right = nx - (xpad - x_shift)

        if ydir < 0:
            y_top = ypad + y_shift
            y_bottom = ny + (y_shift)
        else:
            y_top = y_shift
            y_bottom = ny - (ypad - y_shift)

        assert y_bottom - y_top == ny_og, "something went wrong with the shapes!"
        assert x_right - x_left == nx_og, "something went wrong with the shapes!"

        y_bottoms.append(y_bottom)
        y_tops.append(y_top)
        x_lefts.append(x_left)
        x_rights.append(x_right)
    return n.array(y_tops), n.array(y_bottoms), n.array(x_lefts), n.array(x_rights)


@deprecated("Only used in register_movie(), which is deprecated.")
def get_shifts_3d(im3d, n_procs=12, filter_pcorr=0):
    sims = []
    i = 0
    print(n_procs)
    if n_procs > 1:
        p = Pool(n_procs)
        sims = p.starmap(
            get_shifts_3d_worker,
            [(idx, im3d, filter_pcorr) for idx in range(im3d.shape[0] - 1)],
        )
    else:
        sims = []
        for idx in range(im3d.shape[0] - 1):
            # print(idx)
            sims.append(get_shifts_3d_worker(idx, im3d, filter_pcorr))
    tvecs = n.array([sim["tvec"] for sim in sims])
    tvecs_cum = n.cumsum(tvecs, axis=0)
    return tvecs_cum


@deprecated("Only used in register_movie(), which is deprecated.")
def get_shifts_3d_worker(idx, im3d, filter_pcorr):
    return imreg.similarity(im3d[idx], im3d[idx + 1], filter_pcorr=filter_pcorr)


def gaussian(x, mu, sigma):
    return n.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * n.sqrt(2 * n.pi))


def sum_log_lik_one_line(m, x, y, b=0, sigma_0=10, c=1e-10, m_penalty=0):
    mu = m * x + b
    lik_line = gaussian(y, mu, sigma_0)
    lik = lik_line

    log_lik = n.log(lik + c - m * m_penalty).sum()

    return -log_lik


def calculate_crosstalk_coeff(
    im3d,
    exclude_below=1,
    sigma=0.01,
    peak_width=1,
    verbose=True,
    estimate_gamma=True,
    estimate_from_last_n_planes=None,
    n_proc=1,
    show_plots=True,
    save_plots=None,
    force_positive=True,
    m_penalty=0,
    bounds=None,
    fit_above_percentile=0,
    fig_scale=3,
    n_per_cavity=None,
):
    plt.style.use("seaborn")
    m_opts = []
    m_firsts = []
    all_liks = []
    m_opt_liks = []
    m_first_liks = []
    im3d = im3d.copy()
    if n_per_cavity is None:
        n_per_cavity = im3d.shape[0] // 2
    if force_positive:
        im3d = im3d - im3d.min(axis=(1, 2), keepdims=True)

    ms = n.linspace(0, 1, 101)
    nz, ny, nx = im3d.shape
    # assert im3d.shape[0] == n_per_cavity*2

    if estimate_from_last_n_planes is None:
        estimate_from_last_n_planes = n_per_cavity

    if save_plots is not None:
        plot_dir = os.path.join(save_plots, "crosstalk_plots")
        os.makedirs(plot_dir, exist_ok=True)

    fs = []
    n_plots = estimate_from_last_n_planes
    n_cols = 5
    n_rows = n.ceil(n_plots / n_cols).astype(int)

    # print(n_plots, n_rows, n_cols)
    # print(estimate_from_last_n_planes)
    # f,axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*fig_scale, n_rows*fig_scale))
    # if n_rows == 1: axs = [axs]

    for i in range(estimate_from_last_n_planes):
        # print("Plot for plane %d" % i)
        Y = im3d[nz - i - 1].flatten()
        X = im3d[nz - i - 1 - n_per_cavity].flatten()
        fit_thresh = n.percentile(X, fit_above_percentile)
        # print(fit_thresh)
        idxs = X > n.percentile(X, fit_above_percentile)
        # print(len(idxs), X.shape)

        if n_proc == 1:
            liks = n.array([sum_log_lik_one_line(m, X[idxs], Y[idxs], sigma_0=sigma, m_penalty=m_penalty) for m in ms])
        else:
            p = Pool(n_proc)
            liks = p.starmap(
                sum_log_lik_one_line,
                [(m, X[idxs], Y[idxs], 0, sigma, 1e-10, m_penalty) for m in ms],
            )
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
            print("Plane %d and %d, m_opt: %.2f and m_first: %.2f" % (i, i + n_per_cavity, m_opt, m_first))

        if bounds is None:
            bounds = (0, n.percentile(X, 99.95))
    #     bins = [n.arange(*bounds,1),n.arange(*bounds,1)]
    #     col_id = idx % n_cols
    #     row_id = idx // n_cols
    #     # print(i,idx, col_id, row_id)
    #     ax = axs[row_id][col_id]
    #     ax.set_aspect('equal')
    #     ax.plot(bins[0], m_opt * bins[0], alpha=0.5, linestyle='--')
    #     ax.plot(bins[0], m_first * bins[0], alpha=0.5, linestyle='--')
    #     ax.hist2d(X, Y, bins = bins, norm=colors.LogNorm())
    #     axsins2 = inset_axes(ax, width="30%", height="40%", loc='upper right')
    #     axsins2.grid(False)
    #     axsins2.plot(ms, liks, label='Min: %.2f, 1st: %.2f' % (m_opt, m_first))
    #     # axsins2.set_xlabel("m")
    #     axsins2.set_xticks([m_opt])
    #     axsins2.set_yticks([])
    #     ax.set_xlabel("Plane %d" % i)
    #     ax.set_ylabel("Plane %d" % (i+n_per_cavity))

    # plt.tight_layout()
    # print('showing')
    # if show_plots: plt.show()
    # # print("showed")
    # if save_plots is not None:
    #     print("Saving figure to %s" % plot_dir)
    #     f.savefig(os.path.join(plot_dir, 'plane_fits.png'), dpi=200)
    #     print("saved")
    # else:
    #     plt.show()
    # plt.close()
    # print("Close figure")

    # return
    m_opts = n.array(m_opts)
    m_firsts = n.array(m_firsts)

    best_ms = m_opts[m_opts == m_firsts]
    best_m = best_ms.mean()

    if estimate_gamma:
        gx = gamma.fit(m_opts)
        x = n.linspace(0, 1, 1001)
        gs = gamma.pdf(x, *gx)
        f = plt.figure(figsize=(3, 3))
        plt.hist(m_opts, density=True, log=False, bins=n.arange(0, 1.01, 0.01))
        plt.plot(x, gs)
        plt.yticks([])
        plt.scatter([x[n.argmax(gs)]], [n.max(gs)], label="Best coeff: %.3f" % x[n.argmax(gs)])
        plt.legend()
        plt.xlabel("Coeff value")
        plt.ylabel("")
        plt.xlim(0, 0.4)
        plt.title("Histogram of est. coefficients per plane")
        if save_plots is not None:
            plt.savefig(os.path.join(plot_dir, "gamma_fit.png"), dpi=200)
        if show_plots:
            plt.show()
        plt.close()
        fs.append(f)
        best_m = x[n.argmax(gs)]

    return m_opts, m_firsts, best_m


@deprecated("Only used in register_movie(), which is deprecated.")
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


@deprecated("Not used anywhere in package.")
def register_movie(mov3d, tvecs=None, save_path=None, n_shift_proc=10):

    if tvecs is None:
        im3d = mov3d.mean(axis=1)
        tvecs = get_shifts_3d(im3d, save_path)

    n_planes = mov3d.shape[0]
    shape_mem = mov3d.shape
    size_mem = mov3d.nbytes

    sh_mem = shared_memory.SharedMemory(create=True, size=size_mem)

    mov_reg = n.ndarray(shape_mem, dtype=mov3d.dtype, buffer=sh_mem.buf)
    mov_reg[:] = mov3d[:]

    sh_mem_name = sh_mem.name
    p = Pool(n_shift_proc)

    p.starmap(
        shift_movie_plane,
        [(idx, sh_mem_name, tvecs[idx], shape_mem, mov_reg.dtype) for idx in n.arange(1, n_planes)],
    )

    im3d = mov_reg.mean(axis=1)
    mov_reg_ret = mov_reg.copy()
    sh_mem.close()
    sh_mem.unlink()

    return mov_reg_ret


def create_shmem(shmem_params):
    shmem = shared_memory.SharedMemory(create=True, size=shmem_params["nbytes"])
    shmem_params["name"] = shmem.name
    return shmem, shmem_params


def create_shmem_from_arr(sample_arr, copy=False):
    shmem_params = {
        "dtype": sample_arr.dtype,
        "shape": sample_arr.shape,
        "nbytes": sample_arr.nbytes,
    }
    shmem, shmem_params = create_shmem(shmem_params)
    sh_arr = n.ndarray(shmem_params["shape"], shmem_params["dtype"], buffer=shmem.buf)
    if copy:
        sh_arr[:] = sample_arr[:]
    else:
        sh_arr[:] = 0

    return shmem, shmem_params, sh_arr


def load_shmem(shmem_params):
    shmem = shared_memory.SharedMemory(name=shmem_params["name"], create=False)
    sh_arr = n.ndarray(shmem_params["shape"], shmem_params["dtype"], buffer=shmem.buf)
    return shmem, sh_arr


def close_shmem(shmem_params):
    shmem = shared_memory.SharedMemory(name=shmem_params["name"], create=False)
    shmem.close()


def close_and_unlink_shmem(shmem_params):
    print("Don't use me. I cause memory leaks :(")
    if "name" in shmem_params.keys():
        shmem = shared_memory.SharedMemory(name=shmem_params["name"], create=False)
        shmem.close()
        shmem.unlink()


def get_centroid(ref_img_3d):
    mean_im = ref_img_3d.mean(axis=0)
    M = moments(mean_im, order=1)
    centroid = (int(M[1, 0] / M[0, 0]), int(M[0, 1] / M[0, 0]))
    return centroid


def pad_crop_movie(mov, centroid, crop_size):
    mov = mov[
        :,
        :,
        centroid[0] - crop_size[0] // 2 : centroid[0] + crop_size[0] // 2,
        centroid[1] - crop_size[1] // 2 : centroid[1] + crop_size[1] // 2,
    ]
    nyy, nxx = mov.shape[2:]
    pad = [(0, 0), (0, 0), (0, 0), (0, 0)]
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


def npy_to_dask(files, name="", axis=1, astype=None):
    sample_mov = n.load(files[0], mmap_mode="r")
    file_ts = [n.load(f, mmap_mode="r").shape[axis] for f in files]
    nz, nt_sample, ny, nx = sample_mov.shape

    dtype = sample_mov.dtype
    chunks = [(nz,), (nt_sample,), (ny,), (nx,)]
    chunks[axis] = tuple(file_ts)
    chunks = tuple(chunks)
    name = "from-npy-stack-%s" % name

    keys = list(product([name], *[range(len(c)) for c in chunks]))
    values = [(n.load, files[i], "r") for i in range(len(chunks[axis]))]

    dsk = dict(zip(keys, values))

    arr = darr.Array(dsk, name, chunks, dtype)
    if astype is not None:
        arr = arr.astype(astype)

    return arr


def get_fusing_shifts(raw_img, borders, n_strip=60, x0=0, plot=True, return_ccs=False):
    borders = n.sort(borders)[1:]
    n_border = len(borders)
    nz, ny, nx = raw_img.shape

    best_shifts = n.zeros((nz, n_border))
    cc_maxs = n.zeros((nz, n_border))
    all_ccs = n.zeros((nz, n_border, n_strip))
    for zidx in range(nz):
        for border_idx in range(n_border):
            xx = borders[border_idx]
            lstrip = raw_img[zidx, :, xx - n_strip : xx]
            rstrip = raw_img[zidx, :, xx : xx + n_strip]
            rstrip_norm = rstrip / n.linalg.norm(rstrip, axis=0)
            l0 = lstrip[:, n_strip - 1 - x0]
            l0_norm = l0 / n.linalg.norm(l0)
            cc_full = l0_norm[:, n.newaxis] * rstrip_norm
            cc = cc_full.sum(axis=0)
            all_ccs[zidx, border_idx] = cc
            best_shifts[zidx, border_idx] = cc.argmax()
            cc_maxs[zidx, border_idx] = cc.max()
    if plot:
        plot_fuse_shifts(best_shifts, cc_maxs)
    if return_ccs:
        return best_shifts, cc_maxs, all_ccs
    return best_shifts, cc_maxs


def plot_fuse_shifts(best_shifts, cc_maxs):
    f, axs = plt.subplots(1, 2, figsize=(10, 4))

    ls = axs[0].plot(cc_maxs, color="k", alpha=0.2)
    lx = axs[0].plot(cc_maxs.mean(axis=1), linewidth=3, color="k", label="mean")
    axs[0].legend(ls[:1] + lx, ["individual strips", "mean"])
    axs[0].set_xlabel("Plane #")
    axs[0].set_ylabel("CC between matching columns")

    ls = axs[1].plot(best_shifts, color="k", alpha=0.2)
    lx = axs[1].plot(best_shifts.mean(axis=1), linewidth=3, color="k", label="mean")
    lm = axs[1].axhline(
        int(n.round(n.median(best_shifts))),
        linewidth=2,
        alpha=0.5,
        color="k",
        linestyle="--",
    )
    axs[1].legend(ls[:1] + lx + [lm], ["individual strips", "mean per plane", "mean"])
    axs[1].set_xlabel("Plane #")
    axs[1].set_ylabel("# pix between strips")
    return f


def zscore(x, ax=0, shift=True, scale=True, epsilon=0, keepdims=True):
    m = x.mean(axis=int(ax), keepdims=keepdims)
    std = x.std(axis=int(ax), keepdims=keepdims) + epsilon

    if not scale:
        std = 1
    if not shift:
        m = 0
    return (x - m) / std


def filt(signal, width=3, axis=0, mode="gaussian"):
    if width == 0:
        return signal

    if mode == "gaussian":
        out = gaussian_filter1d(signal, sigma=width, axis=axis)
    else:
        assert False, "mode not implemented"
    return out


def moving_average(x, width=3, causal=True, axis=0, mode="nearest"):
    if width == 1:
        return x
    kernel = n.ones(width * 2 - 1)
    if causal:
        kernel[: int(n.ceil(width / 2))] = 0
    kernel /= kernel.sum()
    return convolve1d(x, kernel, axis=axis, mode=mode)


def get_repo_status(repo_path):
    repo = Repo(repo_path)
    branch = repo.active_branch.name
    is_dirty = repo.is_dirty()
    commit_hash = repo.git.rev_parse("HEAD")
    last_commit = repo.head.commit
    author = last_commit.author.name
    date = last_commit.committed_datetime.strftime("%Y-%d-%m-%H_%M_%S")
    summary = last_commit.summary
    dirty_files = [item.a_path for item in repo.index.diff(None)]

    status = {
        "branch": branch,
        "commit_hash": commit_hash,
        "commit_summary": summary,
        "commit_date": date,
        "commit_author": author,
        "repo_is_dirty": is_dirty,
        "dirty_files": dirty_files,
    }
    return status


def save_benchmark_results(
    results_dir,
    outputs,
    timings,
    repo_status,
    comp_strings=None,
    output_isclose=None,
    is_baseline=False,
):
    if is_baseline:
        dir_name = "baseline"
    else:
        dir_name = datetime.now().strftime("%Y-%d-%m-%H_%M")
    dir_path = os.path.join(results_dir, dir_name)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    n.save(os.path.join(dir_path, "outputs.npy"), outputs)
    n.save(os.path.join(dir_path, "timings.npy"), timings)
    n.save(os.path.join(dir_path, "repo_status.npy"), repo_status)

    if output_isclose is not None:
        n.save(os.path.join(dir_path, "output_isclose.npy"), output_isclose)
    if comp_strings is not None:
        comp_string = ""
        for s in comp_strings:
            comp_string += s
        with open(os.path.join(dir_path, "comp.txt"), "w") as comp_file:
            comp_file.write(comp_string)
    print("Saved benchmark results to %s" % dir_path)
    return dir_path


def load_baseline_results(results_dir):
    outputs = n.load(os.path.join(results_dir, "baseline", "outputs.npy"), allow_pickle=True).item()
    timings = n.load(os.path.join(results_dir, "baseline", "timings.npy"), allow_pickle=True).item()
    repo_status = n.load(os.path.join(results_dir, "baseline", "repo_status.npy"), allow_pickle=True).item()
    return outputs, timings, repo_status


def compare_repo_status(baseline_repo_status, repo_status, print_output=True):
    repo_string = (
        "\
                                    %-20.20s | %-20.20s | \n\
Branch:                             %-20.20s | %-20.20s | \n\
Last commit hash:                   %-20.20s | %-20.20s | \n\
Last commit summ:                   %-20.20s | %-20.20s | \n\
Dirty :                             %-20.20s | %-20.20s | \n\
    "
        % (
            "     Baseline",
            "     Current",
            baseline_repo_status["branch"],
            repo_status["branch"],
            baseline_repo_status["commit_hash"],
            repo_status["commit_hash"],
            baseline_repo_status["commit_summary"],
            repo_status["commit_summary"],
            baseline_repo_status["repo_is_dirty"],
            repo_status["repo_is_dirty"],
        )
    )
    if print_output:
        print(repo_string)
    return repo_string


def compare_timings(baseline_timings, timings, print_output=True):
    timing_keys = baseline_timings.keys()
    timing_string = "Timings (s) \n"
    for key in timing_keys:
        timing_string += "%-36.36s%20.3f | %20.3f | \n" % (
            key,
            baseline_timings[key],
            timings[key],
        )

    if print_output:
        print(timing_string)

    return timing_string


def compare_outputs(baseline_outputs, outputs, rtol=1e-04, atol=1e-06, print_output=True):
    is_closes = {}
    output_keys = baseline_outputs.keys()
    string = "Outputs: \n"
    for key in output_keys:
        base_out = baseline_outputs[key]
        new_out = outputs[key]
        is_close = n.isclose(base_out, new_out, rtol=rtol, atol=atol).flatten()
        if type(base_out) is n.ndarray:
            string += "%-36.36s%-20.20s | %-20.20s |  mismatch: %d / %d (%2.5f %% match) \n" % (
                key,
                " ",
                " ",
                (~is_close).sum(),
                is_close.size,
                100 * (is_close).sum() / is_close.size,
            )
            string += "%-36.36s%-20.20s | %-20.20s | \n" % (
                "           shape: ",
                str(base_out.shape),
                str(new_out.shape),
            )
            string += "%-36.36s%20.3f | %20.3f |\n" % (
                "           mean:",
                base_out.mean(),
                new_out.mean(),
            )
            string += "%-36.36s%20.3f | %20.3f | \n" % (
                "           std:",
                base_out.std(),
                new_out.std(),
            )
        else:
            string += "%-36.36s%20.3f | %20.3f | \n" % (key, base_out, new_out)
        is_closes[key] = is_close
    if print_output:
        print(string)

    return string, is_closes


def benchmark(results_dir, outputs, timings, repo_status):
    baseline_outputs, baseline_timings, baseline_repo_status = load_baseline_results(results_dir)

    repo_comp = compare_repo_status(baseline_repo_status, repo_status)
    timing_comp = compare_timings(baseline_timings, timings)
    output_comp, output_isclose = compare_outputs(baseline_outputs, outputs)

    save_benchmark_results(
        results_dir,
        outputs,
        timings,
        repo_status,
        (repo_comp, timing_comp, output_comp),
        output_isclose,
    )


def to_int(val):
    """
    Properly round a floating point number and return an integer

    Args:
        val (float): number
    """
    return n.round(val).astype(int)


def get_matching_params(param_names, params):
    """
    Return a new dict with only specified keys of params

    Args:
        param_names (list): list of parameter names to keep
        params (dict): dictionary of all parameters

    Returns:
        dict: matching_params, subset of params
    """
    matching_params = {}
    for param_name in param_names:
        matching_params[param_name] = params[param_name]
    return matching_params


def default_log(string, level=0, *args, **kwargs):
    print(("   " * level) + string)


def make_batch_paths(parent_dir, n_batches=1, prefix="batch", suffix="", dirs=True):
    """
    Make n_batches paths within parent_dir to save iteration results.
    By default, it will create parent_dir/batch0001, parent_dir/batch0002, ...

    Args:
        parent_dir (str): Path to parent dir
        n_batches (int, optional): Number of batches. Defaults to 1.
        prefix (str, optional): Prefix of the name of batches. Defaults to 'batch'.
        dirs (bool, optional): If True, make directories. Else, just make pathnames

    Returns:
        _type_: _description_
    """
    batch_dirs = []
    for batch_idx in range(n_batches):
        batch_dir = os.path.join(parent_dir, (prefix + "%04d" + suffix) % batch_idx)
        if dirs:
            os.makedirs(batch_dir, exist_ok=True)
        batch_dirs.append(batch_dir)

    return batch_dirs


# TODO try numba to speed it up? (will have to do seperate cpu/gpu)
def crosstalk_subtract(mov, crosstalk_coeff, cavity_size):
    """
    Subtracts the crosstalk from cavity A in cavity B planes

    Parameters
    ----------
    mov : ndarray
        The inital movie
    crosstalk_coeff : float
        The percentage of crosstalk
    caivty_size : int
        The size of cavity A

    Returns
    -------
    ndarray
        The crosstalk subtracted movie
    """
    nz, __, __, __ = mov.shape
    if nz <= cavity_size:
        return mov
    for i in range(nz - cavity_size):
        mov[i + cavity_size] = mov[i + cavity_size] - crosstalk_coeff * mov[i]
    return mov


def estimate_crosstalk(im3d, cavity_size, step_dist=0.005, max_test_ct=0.5, use_mutual_information=False):
    """
    This function estimates the crosstalk between two cavities by iterativley testing different
    values of the crosstalk and looking at plane2 - ct*plane1 and by either:
    1. Finding the point of inflection of the gradient of the correlation of plane 1 and (plane2 - ct*plane1)
            this can be thought of as when it goes from subtracting crosstalk to adding -ve crosstalk
    2. finding the minimum of mututal information of plane 1 and (plane2 - ct*plane1), this is a overestimate

    Parameters
    ----------
    im3d : ndarray (nz, ny, nx)
        The mean image of the raw data
    cavity_size : int
        The size of the cavity
    step_dist : float, optional
        The step size of tested crosstalk coefficent values, by default 0.005
    max_test_ct : float, optional
        The maximum test corsstalk coefficent value, by default 0.5
    use_mutual_information : bool, optional
        If True will use the mutual information method (not reccomended), by default False

    Returns
    -------
    ndarray, float, dict
        The array of estiamted crosstalk coefficents, the median crosstalk coefficent,
        a dictionary containg the raw ct data
    """
    time_init = time.time()

    nz = im3d.shape[0]  # if you are not ussing all pleans from the second caivty

    if nz <= cavity_size:
        raise ValueError(
            f"Number of z-planes ({nz}) must be >= cavity_size ({cavity_size}). "
            f"Got image with shape {im3d.shape}. "
            "Check init-pass parameters `cavity_size` and `planes`."
        )

    n_second_cavity_planes = nz - cavity_size

    steps = n.arange(0, max_test_ct, step_dist)
    ct_metric = n.zeros((n_second_cavity_planes, steps.shape[0]))
    ct_metric_grad = n.zeros((n_second_cavity_planes, steps.shape[0] - 1))
    d2y = n.zeros((n_second_cavity_planes, steps.shape[0] - 2))
    ct = n.zeros(n_second_cavity_planes)

    for z in range(n_second_cavity_planes):
        for i, step in enumerate(steps):
            base_plane = im3d[z]
            test_plane = im3d[z + cavity_size]

            test_plane_ct_sub = test_plane - step * base_plane

            if use_mutual_information:
                if z == 0 and i == 0:
                    print("Using mutual information, not recommended")
                ct_metric[z, i] = normalized_mutual_information(base_plane, test_plane_ct_sub, bins=100)
            else:
                ct_metric[z, i] = n.corrcoef(base_plane.flatten(), test_plane_ct_sub.flatten())[0, 1]

        if use_mutual_information:
            ct_info = {"ct_metric": ct_metric, "steps": steps}
            ct[z] = steps[n.argmin(ct_metric[z, :])]
        else:
            # The point of inflection of the gradient, is seemingly a good estiamte for the crosstalk
            ct_metric_grad[z, :] = ct_metric[z, 1:] - ct_metric[z, :-1]
            d2y[z, :] = ct_metric[z, :-2] - 2 * ct_metric[z, 1:-1] + ct_metric[z, 2:]
            min_idx = n.argmin(d2y[z, :])

            ct_info = {
                "ct_metric": ct_metric,
                "ct_metric_gradient": ct_metric_grad,
                "ct_metric_d2": d2y,
                "steps": steps,
            }
            ct[z] = steps[min_idx + 2]  # + 2 is to account for the change in shape doing second derivative

    ct_estimate = n.median(ct)
    print(f"Estiamted crosstalk in {time.time() - time_init}s")
    return ct, ct_estimate, ct_info


def plot_ct_hist(crosstalk_planes, show_plots=True, save_plots=None):
    """
    Plots the crsstalk coefficent estiamtes as a histogram

    Parameters
    ----------
    crosstalk_planes : ndarray
        An array of the crosstalk estiamte for each plane
    show_plots : bool, optional
        If True will show the plot, by default True
    save_plots : str, optional
        The path to the save directory ifbeing saved, by default None
    """

    if save_plots is not None:
        plot_dir = os.path.join(save_plots, "crosstalk_plots")
        os.makedirs(plot_dir, exist_ok=True)

    plt.hist(crosstalk_planes, log=False, bins=n.arange(0, 0.5, 0.025))
    plt.axvline(n.median(crosstalk_planes))
    plt.xlabel("Coeff value")
    plt.title("Histogram of est. coefficients per plane")
    if save_plots is not None:
        plt.savefig(os.path.join(plot_dir, "gamma_fit.png"), dpi=200)
    if show_plots:
        plt.show()
    plt.close()


# def ct_gifs(im3d, cavity_size, crosstalk_planes, save_plots=None):
#     """
#     Will aniamte a 3 frame gif, showing the Cavity A palne, the Cavity B plane
#     and the Cavity B - the estiamted crosstalk

#     Parameters
#     ----------
#     im3d : ndarray (nz, ny, nx)
#         The mean image of the raw data
#     cavity_size : int
#         The size of the cavity
#     crosstalk_planes : ndarray
#         An array of the crosstalk estiamte for each plane
#     save_plots : str, optional
#         The path to the save directory ifbeing saved, by default None
#     """
#     if save_plots is not None:
#         plot_dir = os.path.join(save_plots, "crosstalk_plots")
#         os.makedirs(plot_dir, exist_ok=True)
#         plot_gif = os.path.join(plot_dir, "crosstalk.gif")

#     ct_est = n.median(crosstalk_planes)
#     # a median plane
#     # Correct when the number of planes is even, and no planes have the median value!!
#     plane_idx = n.argmin(n.abs(crosstalk_planes - ct_est))

#     # group the origina cavity A plane, cavity B plane and crosstalk sub cavity B plane
#     animate_planes = [
#         im3d[plane_idx],
#         im3d[plane_idx - cavity_size],
#         im3d[plane_idx - cavity_size] - ct_est * im3d[plane_idx],
#     ]
#     animate_planes = n.asarray(animate_planes)

#     titles = ["Cavity A", "Cavity B", "Cavity B crosstalk sub"]
#     tfu.animate_gif(animate_planes, plot_gif, titles=titles)
