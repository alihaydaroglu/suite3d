import os
import numpy as n
import time

from .s2p_registration import nonrigid_transform_data, register_frames
from . import detection3d as det3d
from . import svd_utils as svu
# from . import lbmio
from . import utils
from . import register_gpu as reg_gpu
from . import reg_3d as reg_3d
from . import reference_image as ref
from . import quality_metrics as qm
from .utils import default_log
from .io import s3dio

import traceback
import gc
import threading

try:
    import cupy as cp
except ImportError:
    import numpy as cp


def init_batches(tifs, batch_size, max_tifs_to_analyze=None):
    if max_tifs_to_analyze is not None and max_tifs_to_analyze > 0:
        tifs = tifs[:max_tifs_to_analyze]
    n_tifs = len(tifs)
    n_batches = int(n.ceil(n_tifs / batch_size))

    batches = []
    for i in range(n_batches):
        batches.append(tifs[i * batch_size : (i + 1) * batch_size])

    return batches


def register_mov(
    mov3d,
    refs_and_masks,
    all_ops,
    log_cb=default_log,
    convolve_method="fast_cpu",
    do_rigid=True,
):
    nz, nt, ny, nx = mov3d.shape
    all_offsets = {
        "xmaxs_rr": [],
        "ymaxs_rr": [],
        "cms": [],
        "xmaxs_nr": [],
        "ymaxs_nr": [],
        "cm1s": [],
    }
    for plane_idx in range(nz):
        log_cb("Registering plane %d" % plane_idx, 2)
        mov3d[plane_idx], ym, xm, cm, ym1, xm1, cm1 = register_frames(
            refAndMasks=refs_and_masks[plane_idx],
            frames=mov3d[plane_idx],
            ops=all_ops[plane_idx],
            convolve_method=convolve_method,
            do_rigid=do_rigid,
        )
        all_offsets["xmaxs_rr"].append(xm)
        all_offsets["ymaxs_rr"].append(ym)
        all_offsets["cms"].append(cm)
        all_offsets["xmaxs_nr"].append(xm1)
        all_offsets["ymaxs_nr"].append(ym1)
        all_offsets["cm1s"].append(cm1)

    for k, v in all_offsets.items():
        all_offsets[k] = n.swapaxes(n.array(v), 0, 1)

    return all_offsets


def fuse_movie(mov, n_skip, centers, shift_xs):
    n_skip_l = n_skip // 2
    n_skip_r = n_skip - n_skip_l
    nz, nt, ny, nx = mov.shape

    centers = n.concatenate([centers, [nx]])
    # print(centers)
    n_seams = len(centers)
    nxnew = nx - (n_skip) * (n_seams)
    # print(nxnew)
    mov_fused = n.zeros((nz, nt, ny, nxnew), dtype=mov.dtype)

    # print(mov.shape)
    # print(mov_fused.shape)
    # print(centers)

    for zidx in range(nz):
        curr_x = 0
        curr_x_new = 0
        for i in range(n_seams):
            # print("  Seam %d" % i)
            wid = (centers[i] + shift_xs[zidx]) - curr_x
            # print(wid)
            # print(curr_x, curr_x_new)
            # print(mov_fused[zidx, :, :, curr_x_new: curr_x_new + wid - n_skip].shape)
            # print(wid - n_skip)
            # print(mov[zidx, :, :, curr_x + n_skip_l: curr_x + wid - n_skip_r].shape)
            # print(wid - n_skip_r - n_skip_l)

            target_len = mov_fused[
                zidx, :, :, curr_x_new : curr_x_new + wid - n_skip
            ].shape[-1]
            source_len = mov[
                zidx, :, :, curr_x + n_skip_l : curr_x + wid - n_skip_r
            ].shape[-1]
            source_crop = 0
            if target_len != source_len:
                source_crop = target_len - source_len
                # print("\n\n\n\nXXXXXXXXXXXXXXXCropping source by %d" % source_crop)
            # print(target_len, source_len, source_crop)
            mov_fused[zidx, :, :, curr_x_new : curr_x_new + wid - n_skip] = mov[
                zidx, :, :, curr_x + n_skip_l : curr_x + wid - n_skip_r + source_crop
            ]
            curr_x_new += wid - n_skip
            curr_x += wid

    return mov_fused


def fuse_and_save_reg_file(
    reg_file,
    reg_fused_dir,
    centers,
    shift_xs,
    n_skip,
    crop=None,
    mov=None,
    save=True,
    delete_original=False,
):
    file_name = reg_file.split(os.sep)[-1]
    fused_file_name = os.path.join(reg_fused_dir, "fused_" + file_name)
    if mov is None:
        # print("Loading")
        mov = n.load(reg_file)
        # print("Loaded")

    if crop is not None:
        cz, cy, cx = crop
        mov = mov[cz[0] : cz[1], cy[0] : cy[1], cx[0] : cx[1]]
    mov_fused = fuse_movie(mov, n_skip, centers, shift_xs)

    # if crops is not None:
    # mov_fused = mov_fused[crops[0][0]:crops[0][1], :, crops[1][0]:crops[1][1], crops[2][0]:crops[2][1]]
    if delete_original:
        print("Delelting file: %s" % reg_file)
        os.remove(reg_file)
    if save:
        n.save(fused_file_name, mov_fused)
        return fused_file_name
    else:
        return mov_fused


def fuse_and_save_reg_file_old(
    reg_file,
    reg_fused_dir,
    centers,
    shift_xs,
    nshift,
    nbuf,
    crops=None,
    mov=None,
    save=True,
    delete_original=False,
):
    file_name = reg_file.split(os.sep)[-1]
    fused_file_name = os.path.join(reg_fused_dir, "fused_" + file_name)
    if mov is None:
        print("Loading")
        mov = n.load(reg_file)
        print("Loaded")
    nz, nt, ny, nx = mov.shape
    weights = n.linspace(0, 1, nshift)
    n_seams = len(centers)
    nxnew = nx - (nshift + nbuf) * n_seams
    mov_fused = n.zeros((nz, nt, ny, nxnew), dtype=mov.dtype)
    print("Looping")
    for zidx in range(nz):
        print(zidx)
        curr_x = 0
        curr_x_new = 0
        for i in range(n_seams):
            wid = (centers[i] + shift_xs[zidx]) - curr_x

            mov_fused[zidx, :, :, curr_x_new : curr_x_new + wid - nshift] = mov[
                zidx, :, :, curr_x : curr_x + wid - nshift
            ]
            mov_fused[zidx, :, :, curr_x_new + wid - nshift : curr_x_new + wid] = (
                mov[zidx, :, :, curr_x + wid - nshift : curr_x + wid] * (1 - weights)
            ).astype(n.int16)
            mov_fused[zidx, :, :, curr_x_new + wid - nshift : curr_x_new + wid] += (
                mov[zidx, :, :, curr_x + wid + nbuf : curr_x + wid + nbuf + nshift]
                * (weights)
            ).astype(n.int16)

            curr_x_new += wid
            curr_x += wid + nbuf + nshift
        mov_fused[zidx, :, :, curr_x_new:] = mov[zidx, :, :, curr_x:]
    if crops is not None:
        mov_fused = mov_fused[
            crops[0][0] : crops[0][1],
            :,
            crops[1][0] : crops[1][1],
            crops[2][0] : crops[2][1],
        ]
    if delete_original:
        print("Delelting file: %s" % reg_file)
        os.remove(reg_file)
    if save:
        n.save(fused_file_name, mov_fused)
        return fused_file_name
    else:
        return mov_fused


def init_batch_files(
    job_iter_dir=None,
    job_reg_data_dir=None,
    n_batches=1,
    makedirs=True,
    filename="reg_data",
    dirname="batch",
):
    reg_data_paths = []
    batch_dirs = []
    for batch_idx in range(n_batches):
        if job_reg_data_dir is not None:
            reg_data_filename = filename + "%04d.npy" % batch_idx
            reg_data_path = os.path.join(job_reg_data_dir, reg_data_filename)
            reg_data_paths.append(reg_data_path)
        if makedirs:
            assert job_iter_dir is not None
            batch_dir = os.path.join(job_iter_dir, dirname + "%04d" % batch_idx)
            os.makedirs(batch_dir, exist_ok=True)
            batch_dirs.append(batch_dir)

    return batch_dirs, reg_data_paths

def register_dataset_gpu_from_existing_shifts(
    job,
    tifs,
    params,
    dirs,
    summary,
    existing_summary,
    log_cb=default_log,
    max_gpu_batches=None,
    structural=False,
):
    jobio = s3dio(job)
    
    # Get the registration results from the summary for a 
    # previously registered channel.
    registration_results = job.load_registration_results()
    xmaxs_rr = registration_results["xmaxs_rr"]
    ymaxs_rr = registration_results["ymaxs_rr"]
    xmaxs_nr = registration_results["xmaxs_nr"]
    ymaxs_nr = registration_results["ymaxs_nr"]

    min_pix_vals = summary["min_pix_vals"]
    crosstalk_coeff = summary["crosstalk_coeff"]
    xpad = existing_summary["xpad"]
    ypad = existing_summary["ypad"]
    fuse_shift = existing_summary["fuse_shift"]
    new_xs = existing_summary["new_xs"]
    old_xs = existing_summary["og_xs"]

    # new parameters
    reference_params = summary["reference_params"]
    rmins = reference_params.get("plane_mins", None)
    rmaxs = reference_params.get("plane_maxs", None)
    yblocks, xblocks = reference_params["yblock"], reference_params["xblock"]
    nblocks = reference_params["nblocks"]

    if params["fuse_shift_override"] is not None:
        fuse_shift = params["fuse_shift_override"]
        log_cb("Overriding fuse shift value to %d" % fuse_shift)

    job_reg_data_dir = dirs["registered_fused_data"]

    n_tifs_to_analyze = params.get("total_tifs_to_analyze", len(tifs))
    tif_batch_size = params["tif_batch_size"]
    enforce_positivity = params.get("enforce_positivity", False)
    split_tif_size = params.get("split_tif_size", None)
    n_ch_tif = params.get("n_ch_tif", 30)
    gpu_reg_batchsize = params.get("gpu_reg_batchsize", 10)
    reg_norm_frames = params.get("reg_norm_frames", True)
    cavity_size = params.get("cavity_size", 15)
    save_dtype_str = params.get("save_dtype", "float32")
    nonrigid = params.get("nonrigid", True)
    save_dtype = None
    if save_dtype_str == "float32":
        save_dtype = n.float32
    elif save_dtype_str == "float16":
        save_dtype = n.float16

    # catch if rmins/rmaxs where not calculate in init_pass
    if rmins is None and rmaxs is None:
        log_cb("Not clipping frames for registration")
        rmins = n.array([None for i in range(n_ch_tif)])
        rmaxs = n.array([None for i in range(n_ch_tif)])
    else:
        if not reg_norm_frames:
            log_cb("Not clipping frames for registration")
            rmins = n.array([None for i in range(len(rmins))])
            rmaxs = n.array([None for i in range(len(rmaxs))])

    batches = init_batches(tifs, tif_batch_size, n_tifs_to_analyze)
    n_batches = len(batches)
    reg_data_paths = []

    log_cb(
        "Will analyze %d tifs in %d batches"
        % (len(n.concatenate(batches)), len(batches)),
        0,
    )
    if enforce_positivity:
        log_cb("Enforcing positivity", 1)

    loaded_movs = [n.zeros(1)]

    def io_thread_loader(tifs, batch_idx):
        tic_thread = time.time()
        log_cb("[Thread] Loading batch %d \n" % batch_idx, 5)
        log_cb("   [Thread] Before load %d \n" % batch_idx, 5, log_mem_usage=True)
        loaded_mov = jobio.load_data(tifs, structural=structural)
        loaded_movs[0] = loaded_mov
        log_cb(
            "[Thread] Thread for batch %d ready to join after %2.2f sec \n"
            % (batch_idx, time.time() - tic_thread),
            5,
        )
        log_cb("   [Thread] After load %d \n" % batch_idx, 5, log_mem_usage=True)

    log_cb("Launching IO thread")
    io_thread = threading.Thread(target=io_thread_loader, args=(batches[0], 0))
    io_thread.start()

    file_idx = 0
    for batch_idx in range(n_batches):
        c_ymaxs_rr = ymaxs_rr[batch_idx].T
        c_xmaxs_rr = xmaxs_rr[batch_idx].T
        c_ymaxs_nr = ymaxs_nr[batch_idx]
        c_xmaxs_nr = xmaxs_nr[batch_idx]

        log_cb("Memory at batch %d." % batch_idx, level=3, log_mem_usage=True)
        log_cb("Loading Batch %d of %d" % (batch_idx, n_batches - 1), 0)
        io_thread.join()
        log_cb("Batch %d IO thread joined" % (batch_idx))
        log_cb("Memory after IO thread join", level=3, log_mem_usage=True)

        mov_cpu = loaded_movs[0].copy()
        log_cb("Memory after movie copied from thread", level=3, log_mem_usage=True)
        loaded_movs[0] = n.zeros(1)
        gc.collect()
        log_cb("Memory after thread memory cleared", level=3, log_mem_usage=True)

        if batch_idx + 1 < n_batches:
            log_cb("Launching IO thread for next batch")
            io_thread = threading.Thread(
                target=io_thread_loader, args=(batches[batch_idx + 1], batch_idx + 1)
            )
            io_thread.start()
            log_cb("After IO thread launch:", level=3, log_mem_usage=True)
        nt = mov_cpu.shape[1]
        mov_shifted = []

        mov_shifted = None
        log_cb("Loaded batch of size %s" % ((str(mov_cpu.shape))), 2)
        for gpu_batch_idx in range(int(n.ceil(nt / gpu_reg_batchsize))):
            if max_gpu_batches is not None:
                if gpu_batch_idx >= max_gpu_batches:
                    break
            idx0 = gpu_reg_batchsize * gpu_batch_idx
            idx1 = min(idx0 + gpu_reg_batchsize, nt)
            log_cb("Sending frames %d-%d to GPU for rigid registration" % (idx0, idx1), 3)
            tic_rigid = time.time()

            mov_shifted_gpu = reg_gpu.rigid_2d_reg_gpu_from_existing_shifts(
                mov_cpu[:, idx0:idx1],
                c_ymaxs_rr[:, idx0:idx1],
                c_xmaxs_rr[:, idx0:idx1],
                crosstalk_coeff = crosstalk_coeff, 
                min_pix_vals = min_pix_vals,
                fuse_and_pad=True,
                ypad=ypad, 
                xpad=xpad,
                fuse_shift=fuse_shift,
                new_xs=new_xs,
                old_xs=old_xs,
                cavity_size = cavity_size,
                log_cb=log_cb,
            )
            mov_shifted_cpu = mov_shifted_gpu.get()
            log_cb(
                "Completed rigid registration in %.2f sec" % (time.time() - tic_rigid), 3
            )
            del mov_shifted_gpu

            if mov_shifted is None:
                mov_shifted = n.zeros(
                    (
                        mov_shifted_cpu.shape[1],
                        nt,
                        mov_shifted_cpu.shape[2],
                        mov_shifted_cpu.shape[3],
                    ),
                    n.float32,
                )
                log_cb(
                    "Allocated array of shape %s to store CPU movie"
                    % str(mov_shifted.shape),
                    3,
                )
                log_cb("After array alloc:", level=3, log_mem_usage=True)

            shift_tic = time.time()
            nz = mov_shifted_cpu.shape[1]
            for zidx in range(nz):
                if nonrigid:
                    # print("SHIFITNG: %d" % zidx)
                    # TODO migrate to suite3D?

                    mov_shifted[zidx, idx0:idx1] = nonrigid_transform_data(
                        mov_shifted_cpu[:, zidx],
                        nblocks,
                        xblock=xblocks,
                        yblock=yblocks,
                        ymax1=c_ymaxs_nr[idx0:idx1, zidx],
                        xmax1=c_xmaxs_nr[idx0:idx1, zidx],
                    )
                else:
                    mov_shifted[zidx, idx0:idx1] = mov_shifted_cpu[:, zidx]

            log_cb(
                "Non rigid transformed (on CPU) in %.2f sec" % (time.time() - shift_tic),
                3,
            )

            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

            log_cb("After GPU Batch:", level=3, log_mem_usage=True)
        log_cb("After all GPU Batches:", level=3, log_mem_usage=True)

        if split_tif_size is None:
            split_tif_size = mov_shifted.shape[0]

        for i in range(0, mov_shifted.shape[1], split_tif_size):
            reg_data_path = os.path.join(
                job_reg_data_dir, "fused_reg_data%04d" % file_idx
            )
            if structural:
                reg_data_path += "_structural"
            reg_data_path += ".npy"
            reg_data_paths.append(reg_data_path)
            end_idx = min(mov_shifted.shape[1], i + split_tif_size)
            mov_save = mov_shifted[:, i:end_idx]
            if max_gpu_batches is not None:
                if i > max_gpu_batches * gpu_reg_batchsize:
                    break
            save_t = time.time()
            log_cb(
                "Saving fused, registered file of shape %s to %s"
                % (str(mov_save.shape), reg_data_path),
                2,
            )
            n.save(reg_data_path, mov_save.astype(save_dtype))
            log_cb("Saved in %.2f sec" % (time.time() - save_t), 3)
            file_idx += 1
        log_cb("After full batch saving:", level=3, log_mem_usage=True)

def register_dataset_gpu(
    job, tifs, params, dirs, summary, log_cb=default_log, max_gpu_batches=None
):
    jobio = s3dio(job)

    refs_and_masks = summary["refs_and_masks"]
    ref_img_3d = summary["ref_img_3d"]
    min_pix_vals = summary["min_pix_vals"]
    crosstalk_coeff = summary["crosstalk_coeff"]
    xpad = summary["xpad"]
    ypad = summary["ypad"]
    plane_shifts = summary["plane_shifts"]
    fuse_shift = summary["fuse_shift"]
    new_xs = summary["new_xs"]
    old_xs = summary["og_xs"]

    # new parameters
    reference_params = summary["reference_params"]
    rmins = reference_params.get("plane_mins", None)
    rmaxs = reference_params.get("plane_maxs", None)
    snr_thresh = params.get("snr_thresh", 1.2)  # TODO add values to a default params dictionary
    NRsm = reference_params["NRsm"]
    yblocks, xblocks = reference_params["yblock"], reference_params["xblock"]
    nblocks = reference_params["nblocks"]


    mask_mul, mask_offset, ref_2ds = n.stack([r[:3] for r in refs_and_masks], axis=1)
    mask_mul_nr, mask_offset_nr, ref_nr = n.stack([r[3:] for r in refs_and_masks], axis=1)
    max_shift_nr = 5

    if params["fuse_shift_override"] is not None:
        fuse_shift = params["fuse_shift_override"]
        log_cb("Overriding fuse shift value to %d" % fuse_shift)

    job_iter_dir = dirs["iters"]
    job_reg_data_dir = dirs["registered_fused_data"]

    n_tifs_to_analyze = params.get("total_tifs_to_analyze", len(tifs))
    tif_batch_size = params["tif_batch_size"]
    planes = params["planes"]
    notch_filt = params["notch_filt"]
    enforce_positivity = params.get("enforce_positivity", False)
    mov_dtype = params["dtype"]
    split_tif_size = params.get("split_tif_size", None)
    n_ch_tif = params.get("n_ch_tif", 30)
    max_rigid_shift = params.get("max_rigid_shift_pix", 75)
    gpu_reg_batchsize = params.get("gpu_reg_batchsize", 10)
    max_shift_nr = params.get("max_shift_nr", 3)
    nr_npad = params.get("nr_npad", 3)
    nr_subpixel = params.get("nr_subpixel", 10)
    nr_smooth_iters = params.get("nr_smooth_iters", 2)
    fuse_strips = params.get("fuse_strips", True)
    fix_fastZ = params.get("fix_fastZ", False)
    reg_norm_frames = params.get("reg_norm_frames", True)
    cavity_size = params.get("cavity_size", 15)
    save_dtype_str = params.get("save_dtype", "float32")
    nonrigid = params.get("nonrigid", True)
    save_dtype = None
    if save_dtype_str == "float32":
        save_dtype = n.float32
    elif save_dtype_str == "float16":
        save_dtype = n.float16

    # catch if rmins/rmaxs where not calculate in init_pass
    if rmins is None and rmaxs is None:
        log_cb("Not clipping frames for registration")
        rmins = n.array([None for i in range(n_ch_tif)])
        rmaxs = n.array([None for i in range(n_ch_tif)])
    else:
        if not reg_norm_frames:
            log_cb("Not clipping frames for registration")
            rmins = n.array([None for i in range(len(rmins))])
            rmaxs = n.array([None for i in range(len(rmaxs))])

    if max_rigid_shift < n.ceil(n.max(n.abs(summary["plane_shifts"]))) + 5:
        max_rigid_shift = n.ceil(n.max(n.abs(summary["plane_shifts"]))) + 5

    convert_plane_ids_to_channel_ids = params.get(
        "convert_plane_ids_to_channel_ids", True
    )

    batches = init_batches(tifs, tif_batch_size, n_tifs_to_analyze)
    n_batches = len(batches)
    __, offset_paths = init_batch_files(
        job_iter_dir, job_reg_data_dir, n_batches, makedirs=False, filename="offsets"
    )
    reg_data_paths = []

    log_cb(
        "Will analyze %d tifs in %d batches"
        % (len(n.concatenate(batches)), len(batches)),
        0,
    )
    if enforce_positivity:
        log_cb("Enforcing positivity", 1)

    loaded_movs = [n.zeros(1)]

    def io_thread_loader(tifs, batch_idx):
        tic_thread = time.time()
        log_cb("[Thread] Loading batch %d \n" % batch_idx, 5)
        log_cb("   [Thread] Before load %d \n" % batch_idx, 5, log_mem_usage=True)
        loaded_mov = jobio.load_data(tifs)
        loaded_movs[0] = loaded_mov
        log_cb(
            "[Thread] Thread for batch %d ready to join after %2.2f sec \n"
            % (batch_idx, time.time() - tic_thread),
            5,
        )
        log_cb("   [Thread] After load %d \n" % batch_idx, 5, log_mem_usage=True)

    log_cb("Launching IO thread")
    io_thread = threading.Thread(target=io_thread_loader, args=(batches[0], 0))
    io_thread.start()

    file_idx = 0
    for batch_idx in range(n_batches):
        log_cb("Memory at batch %d." % batch_idx, level=3, log_mem_usage=True)
        offset_path = offset_paths[batch_idx]
        log_cb("Loading Batch %d of %d" % (batch_idx, n_batches - 1), 0)
        io_thread.join()
        log_cb("Batch %d IO thread joined" % (batch_idx))
        log_cb("Memory after IO thread join", level=3, log_mem_usage=True)

        mov_cpu = loaded_movs[0].copy()
        log_cb("Memory after movie copied from thread", level=3, log_mem_usage=True)
        loaded_movs[0] = n.zeros(1)
        gc.collect()
        log_cb("Memory after thread memory cleared", level=3, log_mem_usage=True)

        if batch_idx + 1 < n_batches:
            log_cb("Launching IO thread for next batch")
            io_thread = threading.Thread(
                target=io_thread_loader, args=(batches[batch_idx + 1], batch_idx + 1)
            )
            io_thread.start()
            log_cb("After IO thread launch:", level=3, log_mem_usage=True)
        nt = mov_cpu.shape[1]
        ymaxs_rr = []
        xmaxs_rr = []
        mov_shifted = []
        ymaxs_nr = []
        xmaxs_nr = []

        mov_shifted = None
        # print(mov_cpu.shape)
        log_cb("Loaded batch of size %s" % ((str(mov_cpu.shape))), 2)
        for gpu_batch_idx in range(int(n.ceil(nt / gpu_reg_batchsize))):
            if max_gpu_batches is not None:
                if gpu_batch_idx >= max_gpu_batches:
                    break
            idx0 = gpu_reg_batchsize * gpu_batch_idx
            idx1 = min(idx0 + gpu_reg_batchsize, nt)
            log_cb("Sending frames %d-%d to GPU for rigid registration" % (idx0, idx1), 3)
            tic_rigid = time.time()

            # print("######\n\nBEFORE RIGID: 0.5p: %.3f 99.5p: %.3f, Mean: %.3f, Min: %.3f, Max:%.3f" %
            #        (n.percentile(mov_cpu[10,idx0:idx1],0.5), n.percentile(mov_cpu[10,idx0:idx1],99.5),
            #         mov_cpu[10,idx0:idx1].mean(), mov_cpu[10,idx0:idx1].min(), mov_cpu[10,idx0:idx1].max()))

            mov_shifted_gpu, ymaxs_rr_gpu, xmaxs_rr_gpu, __ = reg_gpu.rigid_2d_reg_gpu(
                mov_cpu[:, idx0:idx1],
                mask_mul,
                mask_offset,
                ref_2ds,
                max_reg_xy=max_rigid_shift,
                min_pix_vals=min_pix_vals,
                rmins=rmins,
                rmaxs=rmaxs,
                crosstalk_coeff=crosstalk_coeff,
                shift=True,
                xpad=xpad,
                ypad=ypad,
                fuse_shift=fuse_shift,
                new_xs=new_xs,
                old_xs=old_xs,
                fuse_and_pad=True,
                cavity_size=cavity_size,
                log_cb=log_cb,
            )

            mov_shifted_cpu = mov_shifted_gpu.get()
            log_cb(
                "Completed rigid registration in %.2f sec" % (time.time() - tic_rigid), 3
            )
            tic_nonrigid = time.time()
            if nonrigid:
                ymaxs_nr_gpu, xmaxs_nr_gpu, snrs = reg_gpu.nonrigid_2d_reg_gpu(
                    mov_shifted_gpu,
                    mask_mul_nr[:, :, 0],
                    mask_offset_nr[:, :, 0],
                    ref_nr[:, :, 0],
                    yblocks,
                    xblocks,
                    snr_thresh,
                    NRsm,
                    rmins,
                    rmaxs,
                    max_shift=max_shift_nr,
                    npad=nr_npad,
                    n_smooth_iters=nr_smooth_iters,
                    subpixel=nr_subpixel,
                    log_cb=log_cb,
                )
                log_cb(
                    "Computed non-rigid shifts in %.2f sec" % (time.time() - tic_rigid), 3
                )

                tic_get = time.time()
                ymaxs_nr_cpu = ymaxs_nr_gpu.get()
                xmaxs_nr_cpu = xmaxs_nr_gpu.get()
            else:
                print("NO NONRIGID\n\n\n")
                tic_get = time.time()
                xmaxs_nr_cpu = n.zeros_like(ymaxs_rr_gpu)
                ymaxs_nr_cpu = n.zeros_like(ymaxs_rr_gpu)

            ymaxs_rr_cpu = ymaxs_rr_gpu.get()
            xmaxs_rr_cpu = xmaxs_rr_gpu.get()
            # print("######\n\nAFter RIGID: 0.5p: %.3f 99.5p: %.3f, Mean: %.3f, Min: %.3f, Max:%.3f" %
            #    (n.percentile(mov_shifted_cpu[:,10],0.5), n.percentile(mov_shifted_cpu[:,10],99.5),
            # mov_shifted_cpu[:,10].mean(), mov_shifted_cpu[:,10].min(),
            # mov_shifted_cpu[:,10].max()))
            # print("SHAPE")
            # print(mov_shifted_cpu.shape)
            del mov_shifted_gpu
            log_cb(
                "Transferred shifted mov of shape %s to CPU in %.2f sec"
                % (str(mov_shifted_cpu.shape), time.time() - tic_get),
                3,
            )

            if mov_shifted is None:
                mov_shifted = n.zeros(
                    (
                        mov_shifted_cpu.shape[1],
                        nt,
                        mov_shifted_cpu.shape[2],
                        mov_shifted_cpu.shape[3],
                    ),
                    n.float32,
                )
                log_cb(
                    "Allocated array of shape %s to store CPU movie"
                    % str(mov_shifted.shape),
                    3,
                )
                log_cb("After array alloc:", level=3, log_mem_usage=True)

            shift_tic = time.time()
            nz = mov_shifted_cpu.shape[1]
            for zidx in range(nz):
                if nonrigid:
                    # print("SHIFITNG: %d" % zidx)
                    # TODO migrate to suite3D?

                    mov_shifted[zidx, idx0:idx1] = nonrigid_transform_data(
                        mov_shifted_cpu[:, zidx],
                        nblocks,
                        xblock=xblocks,
                        yblock=yblocks,
                        ymax1=ymaxs_nr_cpu[:, zidx],
                        xmax1=xmaxs_nr_cpu[:, zidx],
                    )
                else:
                    mov_shifted[zidx, idx0:idx1] = mov_shifted_cpu[:, zidx]

            # print("######\n\nAFter NONRIGID: 0.5p: %.3f 99.5p: %.3f, Mean: %.3f, Min: %.3f, Max:%.3f" %
            #        (n.percentile(mov_shifted[10,idx0:idx1],0.5), n.percentile(mov_shifted[10,idx0:idx1],99.5),
            #         mov_shifted[10,idx0:idx1].mean(), mov_shifted[10,idx0:idx1].min(),
            #         mov_shifted[10,idx0:idx1].max()))
            log_cb(
                "Non rigid transformed (on CPU) in %.2f sec" % (time.time() - shift_tic),
                3,
            )

            # mov_shifted.append(mov_shifted_cpu)
            ymaxs_rr.append(ymaxs_rr_cpu.T)
            xmaxs_rr.append(xmaxs_rr_cpu.T)
            ymaxs_nr.append(ymaxs_nr_cpu)
            xmaxs_nr.append(xmaxs_nr_cpu)

            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

            log_cb("After GPU Batch:", level=3, log_mem_usage=True)

        concat_t = time.time()
        log_cb("Concatenating movie", 2)
        # mov_shifted = mov_shifted_cpu # n.concatenate(mov_shifted,axis=0)
        # print("CONCAT")
        # print(mov_shifted.shape)
        log_cb("Concat in %.2f sec" % (time.time() - concat_t), 3)
        all_offsets = {}
        all_offsets["xmaxs_rr"] = n.concatenate(xmaxs_rr, axis=0)
        all_offsets["ymaxs_rr"] = n.concatenate(ymaxs_rr, axis=0)
        all_offsets["xmaxs_nr"] = n.concatenate(xmaxs_nr, axis=0)
        all_offsets["ymaxs_nr"] = n.concatenate(ymaxs_nr, axis=0)

        log_cb("After all GPU Batches:", level=3, log_mem_usage=True)

        if split_tif_size is None:
            split_tif_size = mov_shifted.shape[0]
        for i in range(0, mov_shifted.shape[1], split_tif_size):
            reg_data_path = os.path.join(
                job_reg_data_dir, "fused_reg_data%04d.npy" % file_idx
            )
            reg_data_paths.append(reg_data_path)
            end_idx = min(mov_shifted.shape[1], i + split_tif_size)
            mov_save = mov_shifted[:, i:end_idx]
            if max_gpu_batches is not None:
                if i > max_gpu_batches * gpu_reg_batchsize:
                    break
            # mov_save = n.swapaxes(mov_save, 0, 1)
            save_t = time.time()
            log_cb(
                "Saving fused, registered file of shape %s to %s"
                % (str(mov_save.shape), reg_data_path),
                2,
            )
            n.save(reg_data_path, mov_save.astype(save_dtype))
            log_cb("Saved in %.2f sec" % (time.time() - save_t), 3)
            file_idx += 1
        n.save(offset_path, all_offsets)

        log_cb("After full batch saving:", level=3, log_mem_usage=True)


def register_dataset_s2p(
    job, tifs, params, dirs, summary, log_cb=default_log, start_batch_idx=0
):
    jobio = s3dio(job)

    ref_img_3d = summary["ref_img_3d"]
    crosstalk_coeff = summary["crosstalk_coeff"]
    refs_and_masks = summary.get("refs_and_masks", None)
    all_ops = summary.get("all_ops", None)
    min_pix_vals = summary["min_pix_vals"]
    fuse_shift = summary["fuse_shift"]
    new_xs = summary["new_xs"]
    old_xs = summary["og_xs"]
    xpad = summary["xpad"]
    ypad = summary["ypad"]

    job_iter_dir = dirs["iters"]
    job_reg_data_dir = dirs["registered_fused_data"]
    n_tifs_to_analyze = params.get("total_tifs_to_analyze", len(tifs))
    tif_batch_size = params["tif_batch_size"]
    planes = params["planes"]
    notch_filt = params["notch_filt"]
    do_subtract_crosstalk = params["subtract_crosstalk"]
    enforce_positivity = params.get("enforce_positivity", False)
    fix_fastZ = params.get("fix_fastZ", False)
    mov_dtype = params["dtype"]
    split_tif_size = params.get("split_tif_size", None)
    n_ch_tif = params.get("n_ch_tif", 30)
    convert_plane_ids_to_channel_ids = params.get(
        "convert_plane_ids_to_channel_ids", True
    )
    cavity_size = params.get("cavity_size", 15)
    nonrigid = params.get("nonrigid", True)
    save_dtype_str = params.get("save_dtype", "float32")
    save_dtype = None
    if save_dtype_str == "float32":
        save_dtype = n.float32
    elif save_dtype_str == "float16":
        save_dtype = n.float16
    reference_params = summary["reference_params"]    
    rmins = reference_params.get("plane_mins", None)
    rmaxs = reference_params.get("plane_maxs", None)
    if all_ops is None:
        all_ops = []
        for i in range(ref_img_3d.shape[0]):
            op = {}
            op['smooth_sigma_time'] = params.get('smooth_sigma_time', 0)
            op['nonrigid'] = params.get('nonrigid', True)
            if rmins is not None and rmaxs is not None:
                op['norm_frames'] = False
                op['rmin'] = rmins[i]
                op['rmax'] = rmaxs[i]
            op['snr_thresh'] = params.get("snr_thresh", 1.2)  
            op['NRsm'] = reference_params["NRsm"]
            op['yblocks'], op['xblocks'] = reference_params["yblock"], reference_params["xblock"]
            op['nblocks'] = reference_params["nblocks"]    
            op['maxregshiftNR'] = params.get("max_shift_nr", 3)
            all_ops.append(op)
        
    batches = init_batches(tifs, tif_batch_size, n_tifs_to_analyze)
    n_batches = len(batches)
    log_cb(
        "Will analyze %d tifs in %d batches"
        % (len(n.concatenate(batches)), len(batches)),
        0,
    )
    if enforce_positivity:
        log_cb("Enforcing positivity", 1)

    # init accumulators
    nz, ny, nx = ref_img_3d.shape
    n_frames_proc = 0

    # __, reg_data_paths = init_batch_files(job_iter_dir, job_reg_data_dir, n_batches, makedirs=False, filename='reg_data')
    reg_data_paths = []
    __, offset_paths = init_batch_files(
        job_iter_dir, job_reg_data_dir, n_batches, makedirs=False, filename="offsets"
    )

    loaded_movs = [0]

    def io_thread_loader(tifs, batch_idx):
        log_cb("   [Thread] Loading batch %d \n" % batch_idx, 5)
        log_cb("   [Thread] Before load %d \n" % batch_idx, 5, log_mem_usage=True)
        loaded_mov = jobio.load_data(tifs)
        # loaded_mov = lbmio.load_and_stitch_tifs(tifs, planes, filt = notch_filt, concat=True,n_ch=n_ch_tif,fix_fastZ=fix_fastZ,
        #                                         convert_plane_ids_to_channel_ids=convert_plane_ids_to_channel_ids, log_cb=log_cb,
        #                                         lbm=params.get('lbm', True), num_colors=params.get('num_colors', None),
        #                                         functional_color_channel=params.get('functional_color_channel', None))
        log_cb("   [Thread] Loaded batch %d \n" % batch_idx, 5)
        log_cb("   [Thread] After load %d \n" % batch_idx, 5, log_mem_usage=True)
        loaded_movs[0] = loaded_mov
        log_cb("   [Thread] Thread for batch %d ready to join \n" % batch_idx, 5)

    log_cb("Launching IO thread")
    io_thread = threading.Thread(
        target=io_thread_loader, args=(batches[start_batch_idx], start_batch_idx)
    )
    io_thread.start()

    file_idx = 0
    for batch_idx in range(start_batch_idx, n_batches):
        try:
            log_cb("Start Batch: ", level=3, log_mem_usage=True)
            # reg_data_path = reg_data_paths[batch_idx]
            offset_path = offset_paths[batch_idx]
            log_cb("Loading Batch %d of %d" % (batch_idx + 1, n_batches), 0)
            io_thread.join()
            log_cb("Batch %d IO thread joined" % (batch_idx))
            log_cb("After IO thread join", level=3, log_mem_usage=True)
            if enforce_positivity:
                # print(loaded_movs[0].shape)
                # print(min_pix_vals.shape)
                log_cb("Subtracting min vals to enfore positivity", 1)
                loaded_movs[0] -= min_pix_vals.reshape(len(min_pix_vals), 1, 1, 1)
                # print(loaded_movs[0].shape
            mov_pad = reg_gpu.fuse_and_pad(
                loaded_movs[0], fuse_shift, ypad, xpad, new_xs, old_xs
            )
            if do_subtract_crosstalk:
                mov_pad = utils.crosstalk_subtract(mov_pad, crosstalk_coeff, cavity_size)
            shmem_mov, shmem_mov_params, mov = utils.create_shmem_from_arr(
                mov_pad, copy=True
            )
            log_cb("After Sharr creation:", level=3, log_mem_usage=True)
            if batch_idx + 1 < n_batches:
                log_cb("Launching IO thread for next batch")
                io_thread = threading.Thread(
                    target=io_thread_loader, args=(batches[batch_idx + 1], batch_idx + 1)
                )
                io_thread.start()
                log_cb("After IO thread launch:", level=3, log_mem_usage=True)
            log_cb("Registering Batch %d" % batch_idx, 1)

            log_cb("Before Reg:", level=3, log_mem_usage=True)
            log_cb()
            all_offsets = register_mov(mov, refs_and_masks, all_ops, log_cb)
            if split_tif_size is None:
                split_tif_size = mov.shape[1]
            for i in range(0, mov.shape[1], split_tif_size):
                reg_data_path = os.path.join(
                    job_reg_data_dir, "fused_reg_data%04d.npy" % file_idx
                )
                reg_data_paths.append(reg_data_path)
                end_idx = min(mov.shape[1], i + split_tif_size)
                log_cb(
                    "Saving registered file of shape %s to %s"
                    % (str(mov[:, i:end_idx].shape), reg_data_path),
                    2,
                )
                n.save(reg_data_path, mov[:, i:end_idx].astype(save_dtype))
                file_idx += 1
            n.save(offset_path, all_offsets)
            log_cb("After reg:", level=3, log_mem_usage=True)

            shmem_mov.close()
            shmem_mov.unlink()
            log_cb("After close + unlink shmem:", level=3, log_mem_usage=True)
            nz, nt, ny, nx = mov.shape
            n_frames_proc_new = n_frames_proc + nt

            n_cleared = gc.collect()
            log_cb("Garbage collected %d items" % n_cleared, 2)
            log_cb("After gc collect: ", level=3, log_mem_usage=True)
        except Exception as exc:
            log_cb("Error occured in iteration %d" % batch_idx, 0)
            tb = traceback.format_exc()
            log_cb(tb, 0)
            break


def calculate_corrmap_from_svd(
    svd_info,
    params,
    dirs,
    log_cb,
    iter_limit=None,
    iter_dir_tag="iters",
    mov_sub_dir_tag="mov_sub",
    svs=None,
    us=None,
):
    assert False, "this is currently broken"
    t_batch_size = params["t_batch_size"]
    temporal_hpf = min(t_batch_size, params["temporal_hpf"])
    if t_batch_size % temporal_hpf != 0:
        temporal_hpf = int(t_batch_size / (n.floor(t_batch_size / temporal_hpf)))
        log_cb(
            "Adjusting temporal hpf to %d to evenly divide %d frames"
            % (temporal_hpf, t_batch_size)
        )
    fix_vmap_edges = params.get("fix_vmap_edges", True)
    do_sdnorm = params.get("do_sdnorm", "True")
    n_proc_corr = params["n_proc_corr"]
    mproc_batchsize = params["mproc_batchsize"]
    sdnorm_exp = params.get("sdnorm_exp", 1.0)

    if mproc_batchsize is None:
        mproc_batchsize = n.ceil(t_batch_size / n_proc_corr)

    npil_filt_size = (
        params["npil_filt_z"],
        params["npil_filt_xy"],
        params["npil_filt_xy"],
    )
    unif_filt_size = (
        params["conv_filt_z"],
        params["conv_filt_xy"],
        params["conv_filt_xy"],
    )

    log_cb(
        "Using conv_filt: %s, %.2f, %.2f"
        % (params["conv_filt_type"], params["conv_filt_z"], params["conv_filt_xy"]),
        1,
    )
    log_cb(
        "Using np_filt: %s, %.2f, %.2f"
        % (params["npil_filt_type"], params["npil_filt_z"], params["npil_filt_xy"]),
        1,
    )
    log_cb("Using normalization exponent of %.2f" % (sdnorm_exp,), 1)

    nz, nt, ny, nx = svd_info["mov_shape"]
    vol_shape = (nz, ny, nx)

    n_batches = int(n.ceil(nt / t_batch_size))
    if iter_limit is not None:
        n_batches = min(iter_limit, n_batches)
        log_cb("Running only %d batches" % n_batches)
    batch_dirs, __ = init_batch_files(
        dirs[iter_dir_tag], makedirs=True, n_batches=n_batches
    )
    __, mov_sub_paths = init_batch_files(
        None,
        dirs[mov_sub_dir_tag],
        makedirs=False,
        n_batches=n_batches,
        filename="mov_sub",
    )
    log_cb("Created files and dirs for %d batches" % n_batches, 1)

    svd_root = "\\".join(svd_info["svd_dirs"][0].split("\\")[:-2])
    log_cb(
        "Will reconstruct SVD movie on-the-fly from %s with %d components"
        % (svd_root, params["n_svd_comp"])
    )
    if svs is None:
        tic = time.time()
        svs = svu.load_and_multiply_stack_svs(
            svd_info["svd_dirs"], params["n_svd_comp"], compute=True
        )
        toc = time.time()
        log_cb(
            "Loaded spatial components in %.2f seconds, %.2f GB"
            % (toc - tic, svs.nbytes / 1024**3)
        )
    else:
        n_comp_sv = svs.shape[0]
        log_cb(
            "Using provided SV matrix, cropping to %d components"
            % int(params["n_svd_comp"])
        )
        if params["n_svd_comp"] > n_comp_sv:
            log_cb(
                "WARNING: the provided SV matrix only has %d components, params specifies %d components!"
                % (n_comp_sv, params["n_svd_comp"])
            )
        svs = svs[:, : int(params["n_svd_comp"])]

    vmap2 = n.zeros((nz, ny, nx))
    mean_img = n.zeros((nz, ny, nx))
    max_img = n.zeros((nz, ny, nx))
    sdmov2 = n.zeros((nz, ny, nx))
    n_frames_proc = 0
    for batch_idx in range(n_batches):
        log_cb("Running batch %d of %d" % (batch_idx + 1, n_batches), 2)
        st_idx = batch_idx * t_batch_size
        end_idx = min(nt, st_idx + t_batch_size)
        n_frames_proc += end_idx - st_idx

        log_cb("Reconstructing from svd", 2)
        recon_tic = time.time()
        if us is not None:
            # print(st_idx, end_idx)
            usx = us[:, st_idx:end_idx, : int(params["n_svd_comp"])]
            log_cb("Using provided U, cropped to %s" % (str(usx.shape)), 3)
        else:
            usx = None
        movx = svu.reconstruct_movie_batch(
            svd_info["svd_dirs"],
            svs,
            (st_idx, end_idx),
            vol_shape,
            svd_info["blocks"],
            us=usx,
            log_cb=log_cb,
        )
        log_cb("Reconstructed in %.2f seconds" % (time.time() - recon_tic), 2)

        log_cb("Calculating corr map", 2)
        corrmap_tic = time.time()
        mov_filt = calculate_corrmap_for_batch(
            movx,
            sdmov2,
            vmap2,
            mean_img,
            max_img,
            temporal_hpf,
            npil_filt_size,
            unif_filt_size,
            params["intensity_thresh"],
            n_frames_proc,
            n_proc_corr,
            mproc_batchsize,
            mov_sub_save_path=mov_sub_paths[batch_idx],
            do_sdnorm=do_sdnorm,
            log_cb=log_cb,
            return_mov_filt=False,
            fix_vmap_edges=fix_vmap_edges,
            sdnorm_exp=sdnorm_exp,
            conv_filt_type=params["conv_filt_type"],
            np_filt_type=params["npil_filt_type"],
            dtype=n.float32,
        )
        log_cb("Calculated corr map in %.2f seconds" % (time.time() - corrmap_tic), 2)

        log_cb("Saving to %s" % batch_dirs[batch_idx], 2)
        n.save(os.path.join(batch_dirs[batch_idx], "vmap2.npy"), vmap2)
        n.save(os.path.join(batch_dirs[batch_idx], "vmap.npy"), vmap2**0.5)
        n.save(os.path.join(batch_dirs[batch_idx], "mean_img.npy"), mean_img)
        n.save(os.path.join(batch_dirs[batch_idx], "max_img.npy"), max_img)
        n.save(os.path.join(batch_dirs[batch_idx], "std2_img.npy"), sdmov2)
        gc.collect()
    vmap = vmap2**0.5
    if fix_vmap_edges and nz > 1:
        vmap[0] = vmap[0] * vmap[1].mean() / vmap[0].mean()
        vmap[-1] = vmap[-1] * vmap[-2].mean() / vmap[-1].mean()
    n.save(os.path.join(batch_dirs[batch_idx], "vmap.npy"), vmap)
    return vmap, mean_img, max_img


# New 3d registration
# TODO tidy up what is needed for 3D case
def register_dataset_gpu_3d(
    job, tifs, params, dirs, summary, log_cb=default_log, max_gpu_batches=None
):
    jobio = s3dio(job)

    refs_and_masks = summary["refs_and_masks"]
    ref_img_3d = summary["ref_img_3d"]
    min_pix_vals = summary["min_pix_vals"]
    crosstalk_coeff = summary["crosstalk_coeff"]
    xpad = summary["xpad"]
    ypad = summary["ypad"]
    plane_shifts = summary["plane_shifts"]
    fuse_shift = summary["fuse_shift"]
    new_xs = summary["new_xs"]
    old_xs = summary["og_xs"]

    # new parameters
    reference_params = summary["reference_params"]
    rmins = reference_params.get("plane_mins", None)
    rmaxs = reference_params.get("plane_maxs", None)
    snr_thresh = 1.2  # TODO add values to a default params dictionary
    NRsm = reference_params["NRsm"]
    yblocks, xblocks = reference_params["yblock"], reference_params["xblock"]
    nblocks = reference_params["nblocks"]
    pc_size = params.get("pc_size", (2, 20, 20))
    frate_hz = params.get("fs", 4)

    # choose the top 2% of pix in each plane to run
    # quality metrics on
    top_pix = qm.choose_top_pix(ref_img_3d)


    # NOTE TODO the current mask_mul etc is uncropped, so currently calculated here but should be changed in reference_image.py
    # when updating to full 3D

    # mask_mul, mask_offset, ref_2ds = n.stack([r[:3] for r in refs_and_masks],axis=1)

    # Current hack to get cropped ref + maks
    sigma = reference_params["sigma"]
    ref_img = ref_img_3d.copy()
    if ypad > 0:
        ref_img = ref_img[:, int(ypad) : int(-ypad)]
    if xpad > 0:
        ref_img = ref_img[:, :, int(xpad) : int(-xpad)]
    # ref_img = ref_img_3d[:, int(ypad):int(-ypad), int(xpad): int(-xpad)]
    mask_mul, mask_offset = ref.compute_masks3D(ref_img, sigma)
    ref_2ds = reg_3d.mask_filter_fft_ref(ref_img, mask_mul, mask_offset, smooth=0.5)

    if params["fuse_shift_override"] is not None:
        fuse_shift = params["fuse_shift_override"]
        log_cb("Overriding fuse shift value to %d" % fuse_shift)

    job_iter_dir = dirs["iters"]
    job_reg_data_dir = dirs["registered_fused_data"]

    n_tifs_to_analyze = params.get("total_tifs_to_analyze", len(tifs))
    tif_batch_size = params["tif_batch_size"]
    planes = params["planes"]
    notch_filt = params["notch_filt"]
    enforce_positivity = params.get("enforce_positivity", False)
    mov_dtype = params["dtype"]
    split_tif_size = params.get("split_tif_size", None)
    n_ch_tif = params.get("n_ch_tif", 30)
    max_rigid_shift = params.get("max_rigid_shift_pix", 75)
    apply_z_shift = params.get("apply_z_shift", False)
    gpu_reg_batchsize = params.get("gpu_reg_batchsize", 10)
    max_shift_nr = params.get("max_shift_nr", 3)
    nr_npad = params.get("nr_npad", 3)
    nr_subpixel = params.get("nr_subpixel", 10)
    nr_smooth_iters = params.get("nr_smooth_iters", 2)
    fuse_strips = params.get("fuse_strips", True)
    fix_fastZ = params.get("fix_fastZ", False)
    reg_norm_frames = params.get("reg_norm_frames", True)
    cavity_size = params.get("cavity_size", 15)
    save_dtype_str = params.get("save_dtype", "float32")
    save_dtype = None
    if save_dtype_str == "float32":
        save_dtype = n.float32
    elif save_dtype_str == "float16":
        save_dtype = n.float16

    # catch if rmins/rmaxs where not calculate in init_pass
    if rmins is None and rmaxs is None:
        log_cb("Not clipping frames for registration")
        rmins = n.array([None for i in range(n_ch_tif)])
        rmaxs = n.array([None for i in range(n_ch_tif)])
    else:
        if not reg_norm_frames:
            log_cb("Not clipping frames for registration")
            rmins = n.array([None for i in range(len(rmins))])
            rmaxs = n.array([None for i in range(len(rmaxs))])

    if max_rigid_shift < n.ceil(n.max(n.abs(summary["plane_shifts"]))) + 5:
        max_rigid_shift = n.ceil(n.max(n.abs(summary["plane_shifts"]))) + 5

    convert_plane_ids_to_channel_ids = params.get(
        "convert_plane_ids_to_channel_ids", True
    )

    batches = init_batches(tifs, tif_batch_size, n_tifs_to_analyze)
    n_batches = len(batches)
    __, offset_paths = init_batch_files(
        job_iter_dir, job_reg_data_dir, n_batches, makedirs=False, filename="offsets"
    )
    reg_data_paths = []

    log_cb(
        "Will analyze %d tifs in %d batches"
        % (len(n.concatenate(batches)), len(batches)),
        0,
    )
    if enforce_positivity:
        log_cb("Enforcing positivity", 1)

    loaded_movs = [n.zeros(1)]

    def io_thread_loader(tifs, batch_idx):
        tic_thread = time.time()
        log_cb("[Thread] Loading batch %d \n" % batch_idx, 5)
        log_cb("   [Thread] Before load %d \n" % batch_idx, 5, log_mem_usage=True)
        loaded_mov = jobio.load_data(tifs)
        # loaded_mov = lbmio.load_and_stitch_tifs(tifs, planes, filt = notch_filt, concat=True,n_ch=n_ch_tif, fix_fastZ=fix_fastZ,
        #                                         convert_plane_ids_to_channel_ids=convert_plane_ids_to_channel_ids, log_cb=log_cb)
        loaded_movs[0] = loaded_mov
        log_cb(
            "[Thread] Thread for batch %d ready to join after %2.2f sec \n"
            % (batch_idx, time.time() - tic_thread),
            5,
        )
        log_cb("   [Thread] After load %d \n" % batch_idx, 5, log_mem_usage=True)
        # log_cb("loaded mov: ")
        # log_cb(str(loaded_mov.shape))

    log_cb("Launching IO thread")
    io_thread = threading.Thread(target=io_thread_loader, args=(batches[0], 0))
    io_thread.start()

    file_idx = 0
    for batch_idx in range(n_batches):
        log_cb("Memory at batch %d." % batch_idx, level=3, log_mem_usage=True)
        offset_path = offset_paths[batch_idx]
        log_cb("Loading Batch %d of %d" % (batch_idx, n_batches - 1), 0)
        io_thread.join()
        log_cb("Batch %d IO thread joined" % (batch_idx))
        log_cb("Memory after IO thread join", level=3, log_mem_usage=True)

        mov_cpu = loaded_movs[0].copy()
        log_cb("Memory after movie copied from thread", level=3, log_mem_usage=True)
        loaded_movs[0] = n.zeros(1)
        gc.collect()
        log_cb("Memory after thread memory cleared", level=3, log_mem_usage=True)

        if batch_idx + 1 < n_batches:
            log_cb("Launching IO thread for next batch")
            io_thread = threading.Thread(
                target=io_thread_loader, args=(batches[batch_idx + 1], batch_idx + 1)
            )
            io_thread.start()
            log_cb("After IO thread launch:", level=3, log_mem_usage=True)
        nt = mov_cpu.shape[1]
        # Change to new kept info
        mov_shifted = []

        mov_shifted = None
        log_cb("Loaded batch of size %s" % ((str(mov_cpu.shape))), 2)
        # New function has loop over batches as part of registration

        time_pre_reg = time.time()
        # log time it takes
        phase_corr_shifted, int_shift, pc_peak_loc, sub_pixel_shifts, mov_cpu = (
            reg_3d.rigid_3d_ref_gpu(
                mov_cpu,
                mask_mul,
                mask_offset,
                ref_2ds,
                pc_size,
                batch_size=gpu_reg_batchsize,  # TODO make xpad/ypad automatically integers
                rmins=rmins,
                rmaxs=rmaxs,
                crosstalk_coeff=crosstalk_coeff,
                shift_reg=False,
                xpad=int(xpad),
                ypad=int(ypad),
                fuse_shift=fuse_shift,
                new_xs=new_xs,
                old_xs=old_xs,
                plane_shifts=plane_shifts,
                process_mov=True,
                cavity_size=cavity_size,
            )
        )

        log_cb(f"Completed rigid reg on batch in :{time.time() - time_pre_reg}s")

        time_shift = time.time()
        # shift entire abtch on cpu at once
        # log this info
        mov_shifted = reg_3d.shift_mov_fast(mov_cpu, -int_shift)

        if apply_z_shift:
            # if there is at least one 
            if n.max(int_shift[0]) > 1:
                mov_shifted = reg_3d.shift_mov_z(mov_shifted, int_shift)
        log_cb(f"Shifted the mov in: {time.time() - time_shift}s")

        # NOTE changed this so gets int_shifts + sub_pixel shifts etc
        all_offsets = {}
        all_offsets["phase_corr_shifted"] = phase_corr_shifted
        all_offsets["int_shift"] = int_shift
        all_offsets["pc_peak_loc"] = pc_peak_loc
        all_offsets["sub_pixel_shifts"] = sub_pixel_shifts

        log_cb("After all GPU Batches:", level=3, log_mem_usage=True)

        if split_tif_size is None:
            split_tif_size = mov_shifted.shape[0]
        for i in range(0, mov_shifted.shape[1], split_tif_size):
            reg_data_path = os.path.join(
                job_reg_data_dir, "fused_reg_data%04d.npy" % file_idx
            )
            reg_data_paths.append(reg_data_path)
            end_idx = min(mov_shifted.shape[1], i + split_tif_size)
            mov_save = mov_shifted[:, i:end_idx]
            if max_gpu_batches is not None:
                if i > max_gpu_batches * gpu_reg_batchsize:
                    break
            # mov_save = n.swapaxes(mov_save, 0, 1)
            save_t = time.time()
            log_cb(
                "Saving fused, registered file of shape %s to %s"
                % (str(mov_save.shape), reg_data_path),
                2,
            )
            n.save(reg_data_path, mov_save.astype(save_dtype))
            log_cb("Saved in %.2f sec" % (time.time() - save_t), 3)

            metrics_path = os.path.join(
                job_reg_data_dir, "reg_metrics_%04d.npy" % file_idx
            )
            mean_img_path = os.path.join(job_reg_data_dir, "mean_img_%04d.npy" % file_idx)
            log_cb("Computing quality metrics and saving", 2)

            mean_img, metrics = qm.compute_metrics_for_movie(
                mov_save, frate_hz, top_pix=top_pix
            )
            n.save(mean_img_path, mean_img)
            n.save(metrics_path, metrics)

            file_idx += 1
        n.save(offset_path, all_offsets)

        log_cb("After full batch saving:", level=3, log_mem_usage=True)
