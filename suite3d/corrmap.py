import os
import numpy as n
import gc
from dask import array as darr
from . import default_params
from . import detection3d as dtu
from .utils import to_int, default_log, get_matching_params, make_batch_paths
from . import utils
from . import extension as ext
import time
import multiprocessing

corr_map_param_names = [
    "voxel_size_um",
    "temporal_hpf",
    "edge_crop_npix",
    "npil_filt_type",
    "npil_filt_z_um",
    "npil_filt_xy_um",
    "cell_filt_type",
    "cell_filt_z_um",
    "cell_filt_xy_um",
    "fix_vmap_edge_planes",
    "detection_timebin",
    "sdnorm_exp",
    "intensity_thresh",
    "standard_vmap",
]

computation_param_names = ["n_proc", "dtype", "t_batch_size"]


def calculate_corrmap(
    mov,
    params,
    summary=None,
    batch_dir=None,
    save_mov_sub=True,
    mov_sub_dir=None,
    iter_limit=None,
    log=default_log,
):
    """
    Compute the correlation map of a large movie in batches, and save results to disk

    Args:
        mov (ndarray or dask array): nz,nt,ny,nx. Can be very big, dask recommended
        params (dict): Dictionary of parameters containing the keys in the lists above
        summary (dict, optional): Output of job.load_summary(). Only necessary if edge_crop_npix > 0, because we need to know the plane shifts to crop the edges of planes. Defaults to None.
        batch_dir (str, optional): Path to directory where batch results should be stored. Defaults to None.
        mov_sub_dir (str, optional): Path to directory where mov_sub should be stored. Defaults to None.
        iter_limit (int, optional): Number of batches to do. Defaults to None.
        log (func, optional): Defaults to default_log.

    Returns:
        _type_: _description_
    """
    nz, nt, ny, nx = mov.shape
    log(
        "Computing correlation map of movie with %d frames, volume shape: %d, %d, %d"
        % (nt, nz, ny, nx),
        1,
    )

    # get two sub-dictionaries of params with relevant parameters
    corr_map_params = get_matching_params(corr_map_param_names, params)
    computation_params = get_matching_params(computation_param_names, params)
    save_dtype_str = params.get("save_dtype", "float32")
    save_dtype = None
    if save_dtype_str == "float32":
        save_dtype = n.float32
    elif save_dtype_str == "float16":
        save_dtype = n.float16

    # compute how many batches we need to split the movie into
    t_batch_size = computation_params["t_batch_size"]
    dtype = computation_params["dtype"]
    n_batches = to_int(n.ceil(nt / t_batch_size))
    if iter_limit is not None:
        n_batches = min(iter_limit, n_batches)

    # prepare the directories to save results into
    if dir is None:
        save = False
        batch_dirs = None
        mov_sub_paths = None
    else:
        # make a set of directories to store intermediate results,
        # and a set of filenames where mov_sub will be saved
        save = True
        batch_dirs = make_batch_paths(batch_dir, n_batches, prefix="batch", dirs=True)
        mov_sub_paths = make_batch_paths(
            mov_sub_dir, n_batches, prefix="mov_sub", suffix=".npy", dirs=False
        )

    # initialize accumulators
    accums = init_corr_map_accumulators((nz, ny, nx), dtype=dtype)

    for batch_idx in range(n_batches):
        log("prep", tic=True)
        start_idx = batch_idx * t_batch_size
        end_idx = min(nt, start_idx + t_batch_size)
        log("Running batch %d of %d" % (batch_idx + 1, n_batches), 2)
        mov_batch = mov[:, start_idx:end_idx]

        log("batch_timebin", tic=True)
        
        if corr_map_params.get("detection_timebin", 1) > 1:
            log(
                f"Binning with timebin of size {corr_map_params['detection_timebin']:02d}",
                2,
            )
            mov_batch = ext.binned_mean_ax1(
                mov_batch, corr_map_params["detection_timebin"]
            )
        log("batch_timebin", toc=True)
        # change the order to nt, nz, ny, nx
        # first try doing in dask, because mov is probably a dask array
        # if that fails (if mov is not a dask array), do it in numpy
        try:
            mov_batch = darr.swapaxes(mov_batch, 0, 1).compute().astype(dtype)
        except:
            log("Not a dask array", 3)
            mov_batch = n.swapaxes(mov_batch, 0, 1).astype(dtype)
        # compute the correlation map for this batch and update accumulators
        log("prep", toc=True)
        log("batch", tic=True)

        n_processors = computation_params["n_proc"]
        pool = multiprocessing.Pool(n_processors)
        vmap_batch, mov_sub_batch = compute_corr_map_batch(
            mov_batch,
            corr_map_params,
            computation_params,
            accums,
            summary,
            log,
            pool=pool,
        )
        log("batch", toc=True)
        log("save", tic=True)
        if save:
            # save results to previously created dirs
            save_batch_results(
                vmap_batch,
                accums,
                batch_dirs[batch_idx],
            )
            if save_mov_sub:
                n.save(mov_sub_paths[batch_idx], mov_sub_batch.astype(save_dtype))
        log("save", toc=True)
    gc.collect()
    save_batch_results(vmap_batch, accums, batch_dir)
    return vmap_batch


def compute_corr_map_batch(
    mov,
    corr_map_params=None,
    computation_params=None,
    accum=None,
    summary=None,
    log=default_log,
    pool=None,
):
    # TODO DOCSTRING
    log("batch_setup", tic=True)
    # get the size of the movie
    nb, nz, ny, nx = mov.shape

    # if parameter dictionaries are not provided, load the default
    # parameter values from default_params.npy
    if corr_map_params is None:
        corr_map_params = default_params.get_matching_default_params(corr_map_param_names)
    if computation_params is None:
        computation_params = default_params.get_matching_default_params(
            computation_param_names
        )

    # load relevant parameters, and convert micron-based params to pixels
    # these parameters relate to algorithm. see default_params.npy for descriptions
    vz, vy, vx = corr_map_params["voxel_size_um"]
    temporal_hpf = to_int(corr_map_params["temporal_hpf"])
    temporal_hpf = min(nb, temporal_hpf)
    if nb % temporal_hpf != 0:
        temporal_hpf = int(nb / (n.floor(nb / temporal_hpf)))
        log(
            "Adjusting temporal hpf to %d to evenly divide %d frames"
            % (temporal_hpf, nb),
            4,
        )

    npil_filt_type = corr_map_params["npil_filt_type"]
    edge_crop_npix = corr_map_params["edge_crop_npix"]
    if edge_crop_npix is not None and edge_crop_npix > 0:
        assert summary is not None
    npil_filt_pix = (
        (corr_map_params["npil_filt_z_um"] / vz),
        (corr_map_params["npil_filt_xy_um"] / vy),
        (corr_map_params["npil_filt_xy_um"] / vx),
    )
    cell_filt_type = corr_map_params["cell_filt_type"]
    cell_filt_pix = (
        (corr_map_params["cell_filt_z_um"] / vz),
        (corr_map_params["cell_filt_xy_um"] / vy),
        (corr_map_params["cell_filt_xy_um"] / vx),
    )
    sdnorm_exp = corr_map_params["sdnorm_exp"]
    intensity_thresh = corr_map_params["intensity_thresh"]
    # fix_vmap_edge_planes = corr_map_params["fix_vmap_edge_planes"]

    # these parameters relate to computational resources
    n_processors = computation_params["n_proc"]
    dtype = computation_params["dtype"]
    minibatch_size = max(20, int(n.ceil(nb / n_processors)))

    # make sure mov is the correct dtype
    if mov.dtype is not dtype:
        mov = mov.astype(dtype)

    # if calling the function repeatedly with differnt batches,
    # results will accumulate in 'accumulators'. If this call is the
    # first (or only) batch, then we want to initialize accumulators
    if accum is None:
        log("Initializing correlation map accumulators", 2)
        accum = init_corr_map_accumulators((nz, ny, nx), dtype)

    # nb is the number of frames in the batch we are currently processing
    # nt is the number of frames processed in previous batches
    nt = accum["n_frames_proc"]

    log("batch_setup", toc=True)
    #### correlation map algorithm #####

    # set the edges of each plane to 0. Otherwise registration causes artifacts
    log("batch_edgecrop", tic=True)
    mov = utils.edge_crop_movie(mov, summary, edge_crop_npix)
    log("batch_edgecrop", toc=True)

    log("accum_meanmeax", tic=True)
    # add the frames from the current batch to accumulated mean,max
    dtu.accumulate_mean(accum["mean_vol"], mov, nt)
    dtu.accumulate_max(accum["max_vol"], mov)
    log("accum_meanmeax", toc=True)
    # a simple high-pass filter by subtracting the rolling mean

    log("batch_rolling_mean_filt", tic=True)
    mov = dtu.hp_rolling_mean_filter(mov, temporal_hpf, copy=False)
    log("batch_rolling_mean_filt", toc=True)
    # compute the standard deviation of temporal differences for each voxel
    # (e.g. the "peakiness" of each voxel) and normalize the movie by it

    log("batch_accum_sdmov", tic=True)
    log("THIS IS A BOTTLENECK - parallelize", 4)
    sdmov = dtu.accumulate_sdmov(
        accum["sdmov_2"],
        mov,
        nt,
    )
    log("batch_accum_sdmov", toc=True)
    log("batch_norm_sdmov", tic=True)
    # log(f"Normalizing with sdnorm {sdnorm_exp}")
    # log(f"SDMOV std {sdmov.std()}, mean {sdmov.mean()}, type {sdmov.dtype}")
    # log(f"mov mean: {mov.mean()}")
    mov = dtu.normalize_movie_by_sdmov(mov, sdmov, sdnorm_exp)

    # log(f"mov mean: {mov.mean()}")
    log("batch_norm_sdmov", toc=True)

    log("batch_filt_reduce", tic=True)
    vmap_2, mov_sub = dtu.filter_and_reduce_movie(
        mov,
        npil_filt_type,
        npil_filt_pix,
        cell_filt_type,
        cell_filt_pix,
        intensity_thresh,
        n_processors,
        minibatch_size,
        standard_vmap=corr_map_params["standard_vmap"],
        log=log,
        pool=pool,
    )

    log("batch_filt_reduce", toc=True)

    log("batch_accum_vmap", tic=True)
    vmap = dtu.accumulate_vmap_2(accum["vmap_2"], vmap_2, nt + nb)
    log("batch_accum_vmap", toc=True)
    accum["n_frames_proc"] += nb
    # log(f"VMAP std {vmap.std()}, mean {vmap.mean()}, type {vmap.dtype}")

    return vmap, mov_sub


def init_corr_map_accumulators(vol_shape, dtype=n.float32):
    """
    Initialize the accumulators used in the computation of correlation maps.
    When the computation is done in batches, we need to accumulate some values
    across batches, which are stored in the dictionary constructed here

    Args:
        vol_shape (tuple): nz,ny,nx shape of the volume in integer pixels
        dtype (dtype, optional): Type for the accumulators created. Defaults to n.float32.
    """
    accumulators = {}
    accumulators["mean_vol"] = n.zeros(vol_shape, dtype=dtype)
    accumulators["max_vol"] = n.zeros(vol_shape, dtype=dtype)
    accumulators["vmap_2"] = n.zeros(vol_shape, dtype=dtype)
    accumulators["sdmov_2"] = n.zeros(vol_shape, dtype=dtype)
    accumulators["n_frames_proc"] = 0
    return accumulators


def save_batch_results(vmap_batch, accums, batch_dir):
    n.save(os.path.join(batch_dir, "vmap2.npy"), accums["vmap_2"])
    n.save(os.path.join(batch_dir, "vmap.npy"), vmap_batch)
    n.save(os.path.join(batch_dir, "mean_img.npy"), accums["mean_vol"])
    n.save(os.path.join(batch_dir, "max_img.npy"), accums["max_vol"])
    n.save(os.path.join(batch_dir, "std2_img.npy"), accums["sdmov_2"])
