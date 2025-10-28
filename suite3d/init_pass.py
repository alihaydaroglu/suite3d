import os
import numpy as n

from . import utils
# from . import lbmio
from . import reference_image as ref
from .utils import default_log
from .developer import todo, deprecated

from .io import s3dio


def choose_init_tifs(
    tifs, n_init_files, init_file_pool_lims=None, method="even", seed=2358
):
    init_file_pool = []
    if init_file_pool_lims is not None:
        for limits in init_file_pool_lims:
            init_file_pool += tifs[limits[0] : limits[1]]
    else:
        init_file_pool = tifs
    init_file_pool = n.array(init_file_pool)

    if method == "even":
        sample_file_ids = n.linspace(0, len(init_file_pool), n_init_files + 2, dtype=int)[1:-1]
        sample_tifs = n.array(init_file_pool)[sample_file_ids]
    elif method == "random":
        # n.random.seed(seed)
        sample_tifs = n.random.choice(init_file_pool, n_init_files, replace=False)

    return sample_tifs


@deprecated("Using s3dio.load_data instead")
def load_init_tifs(
    init_tifs,
    planes,
    filter_params,
    n_ch_tif=30,
    convert_plane_ids_to_channel_ids=True,
    fix_fastZ=False,
    log_cb=default_log,
    lbm=True,
    num_colors=None,
    functional_color_channel=None,
):
    full_mov = lbmio.load_and_stitch_tifs(
        init_tifs,
        planes=planes,
        convert_plane_ids_to_channel_ids=convert_plane_ids_to_channel_ids,
        n_ch=n_ch_tif,
        filt=filter_params,
        concat=False,
        fix_fastZ=fix_fastZ,
        log_cb=log_cb,
        lbm=lbm,
        num_colors=num_colors,
        functional_color_channel=functional_color_channel,
    )
    full_mov = n.concatenate(full_mov, axis=1)
    return full_mov


def run_init_pass(job, structural=False):
    tifs = job.tifs
    params = job.params

    jobio = s3dio(job)

    summary_name = "summary_structural.npy" if structural else "summary.npy"
    summary_path = os.path.join(job.dirs["summary"], summary_name)
    job.log("Saving summary to %s" % summary_path, 0)
    if not os.path.isdir(job.dirs["summary"]):
        job.log("Summary dir does not exist!!")
        raise ValueError("Summary dir does not exist!!")

    init_tifs = choose_init_tifs(
        tifs,
        params["n_init_files"],
        params["init_file_pool"],
        params["init_file_sample_method"],
    )
    n_ch_tif = job.params.get("n_ch_tif", 30)
    job.log("Loading init tifs with %d channels" % n_ch_tif)
    init_mov = jobio.load_data(init_tifs, structural=structural)

    nz, nt, ny, nx = init_mov.shape
    if params["init_n_frames"] is not None:
        if nt < params["init_n_frames"]:
            job.log(
                "Not enough frames in loaded tifs - using %d init frames instead" % nt
            )
        elif nt > params["init_n_frames"]:
            subset_ts = n.random.choice(
                n.arange(nt), params["init_n_frames"], replace=False
            )
            job.log(
                "Selecting %d random frames from the init tif files"
                % params["init_n_frames"]
            )
            init_mov = init_mov[:, subset_ts]
                
    nz, nt, ny, nx = init_mov.shape
    job.log("Loaded movie with %d frames and shape %d, %d, %d" % (nt, nz, ny, nx))
    im3d = init_mov.mean(axis=1)
    im3d_raw = im3d.copy()
    if job.params.get("enforce_positivity", False):
        # min_pix_vals = init_mov.min(axis=(1, 2, 3), keepdims=True)[:,0].astype(int)
        # min_pix_vals = n.percentile(init_mov.reshape(nz, -1), 1, axis=1).astype(int)
        min_pix_vals = im3d.min(axis=(1, 2), keepdims=False).astype(int)
        job.log("Enforcing positivity in mean image", 2)
        init_mov -= min_pix_vals[:, n.newaxis, n.newaxis, n.newaxis]
        im3d -= min_pix_vals[:, n.newaxis, n.newaxis]

        # job.log("Min pix vals: %s" % str(min_pix_vals.flatten()), 3)
    else:
        min_pix_vals = None
    if params["subtract_crosstalk"] and params["lbm"]:
        if params["override_crosstalk"] is not None:
            cross_coeff = params["override_crosstalk"]
            job.log("Subtracting crosstalk with forced coefficient %0.3f" % cross_coeff)
        else:
            # TODO delete when happy with new crosstalk method
            # __, __, cross_coeff = utils.calculate_crosstalk_coeff(im3d,
            #                                         estimate_from_last_n_planes=params['crosstalk_n_planes'],
            #                                         sigma=params['crosstalk_sigma'], fit_above_percentile = params['crosstalk_percentile'],
            #                                         show_plots=True, save_plots=job.dirs['summary'],
            #                                         verbose=(job.verbosity == 2))

            crosstalk_planes, cross_coeff, ct_info = utils.estimate_crosstalk(
                im3d, job.params["cavity_size"]
            )
            # utils.plot_ct_hist(
            #     crosstalk_planes, show_plots=True, save_plots=job.dirs["summary"]
            # )
            # utils.ct_gifs(
            #     im3d,
            #     job.params["cavity_size"],
            #     crosstalk_planes,
            #     save_plots=job.dirs["summary"],
            # )

            job.log("Subtracting with estimated coefficient %0.3f" % cross_coeff)
            if cross_coeff > 0.4 or cross_coeff < 0.01:
                job.log("WARNING - seems like coefficient esimation failed!")
        for plane in range(len(params["planes"])):
            if plane >= params["cavity_size"] and plane - params["cavity_size"] >= 0:
                # plane_idx = n.where(n.array(params['planes']) == plane)[0][0]
                # sub_plane_idx = n.where(n.array(params['planes']) == plane - 15)[0][0]

                job.log(
                    "Subtracting plane %d from %d"
                    % (plane - params["cavity_size"], plane),
                    3,
                )
                # job.log("        Corresponds to index %d from %d" % (sub_plane_idx, plane_idx))
                init_mov[plane] = (
                    init_mov[plane]
                    - init_mov[plane - params["cavity_size"]] * cross_coeff
                )
        im3d = init_mov.mean(axis=1)
    else:
        job.log("No crosstalk estimation or subtraction")
        crosstalk_planes = None
        cross_coeff = None
        ct_info = None

    if job.params.get("fuse_strips", True):
        xs = jobio.load_roi_start_pix()[1]
        if job.params.get("fuse_shift_override", None) is not None:
            fuse_shift = int(job.params["fuse_shift_override"])
            fuse_shifts = None
            fuse_ccs = None
            # job.log("Overriding", 2)
        else:
            job.log("Estimating fusing shifts")
            fuse_shifts, fuse_ccs = utils.get_fusing_shifts(im3d_raw, xs)
            fuse_shift = int(n.round(n.median(fuse_shifts)))
            job.log("Using best fuse shift of %d" % fuse_shift)
    else:
        fuse_shift = 0
        fuse_shifts = None
        fuse_ccs = None
        xs = None
    # return

    reference_params = {
        "percent_contribute": params.get("percent_contribute", 0.9),
        "block_size": params.get("block_size", [128, 128]),
        "sigma": params.get("sigma_reference", [1.45, 0]),  # Y/X smooth, Z smooth
        "smooth_sigma": params.get("smooth_sigma_reference", 1.15),  # spatial taper width
        "niter": params.get("n_reference_iterations", 8),
        "max_reg_xy_reference": params.get("max_reg_xy_reference", 50),
        "pc_size": params.get(
            "pc_size", (2, 20, 20)
        ),  # the max size examined in registration
        "batch_size": params.get("gpu_reference_batch_size", 20),  # keep in gpu RAM
        "3d_reg": params.get("3d_reg", True),  # Default is true
        "plane_to_plane_alignment": params.get(
            "plane_to_plane_alignment", True
        ),  # whether to align planes to each other
    }
    if job.params.get("fuse_strips", True):
        mov_fuse, new_xs, og_xs = ref.fuse_mov(init_mov, fuse_shift, n.sort(xs))
    else:
        mov_fuse = init_mov
        new_xs = [[0, mov_fuse.shape[3]]]
        og_xs = [[0, mov_fuse.shape[3]]]

    if reference_params["3d_reg"]:
        job.log("Using 3d registration")
        (
            tvecs,
            ref_image,
            all_refs_and_masks,
            pad_sizes,
            reference_params,
            reference_info,
            shifted_mov,
        ) = ref.compute_reference_and_masks_3d(
            mov_fuse,
            reference_params,
            log_cb=job.log,
            use_GPU=params.get("gpu_reg", True),
        )

        init_mov_name = "init_mov_structural.npy" if structural else "init_mov.npy"
        summary_mov_path = os.path.join(job.dirs["summary"], init_mov_name)
        n.save(summary_mov_path, shifted_mov)
        job.log("Saved init mov to %s" % summary_mov_path)

        summary = {
            "ref_img_3d": ref_image,  # ctalk-sub and padded and plane-shifted
            "raw_img": im3d_raw,  # right from the tiff file
            "img": im3d,  # crosstalk-subtracted
            "crosstalk_coeff": cross_coeff,
            "crosstalk_planes": crosstalk_planes,
            "crosstalk_info": ct_info,  # dictionary of ct data for each test crosstalk and plane
            "plane_shifts": tvecs,
            "refs_and_masks": all_refs_and_masks,
            "reference_params": reference_params,
            "reference_info": reference_info,
            "min_pix_vals": min_pix_vals,
            "fuse_shifts": fuse_shifts,
            "fuse_shift": fuse_shift,
            "fuse_ccs": fuse_ccs,
            "tiffile_xs": xs,
            "xpad": pad_sizes[0],
            "ypad": pad_sizes[1],
            "new_xs": new_xs,
            "og_xs": og_xs,
            "init_mov_path": summary_mov_path,
            "init_tifs": init_tifs,
        }
    else:
        job.log("Using 2d registration")
        tvecs, ref_image, ref_padded, all_refs_and_masks, pad_sizes, reference_params = (
            ref.compute_reference_and_masks(
                mov_fuse,
                reference_params,
                log_cb=job.log,
                use_GPU=params.get("gpu_reg", True),
            )
        )
        summary = {
            "ref_img_3d": ref_image,  # ctalk-sub and padded and plane-shifted
            "ref_img_3d_unaligned": ref_padded,  # ctalk-sub and padded, 3d doesnt have unalligned reference img
            "raw_img": im3d_raw,  # right from the tiff file
            "img": im3d,  # crosstalk-subtracted
            "crosstalk_coeff": cross_coeff,
            "crosstalk_planes": crosstalk_planes,
            "crosstalk_info": ct_info,  # dictionary of ct data for each test crosstalk and plane
            "plane_shifts": tvecs[0],
            "plane_shifts_uncorrected": tvecs[1],
            "refs_and_masks": all_refs_and_masks,
            "reference_params": reference_params,
            "min_pix_vals": min_pix_vals,
            "fuse_shifts": fuse_shifts,
            "fuse_shift": fuse_shift,
            "fuse_ccs": fuse_ccs,
            "tiffile_xs": xs,
            "xpad": pad_sizes[0],
            "ypad": pad_sizes[1],
            "new_xs": new_xs,
            "og_xs": og_xs,
            "init_tifs": init_tifs,
        }
    
    job.log("Saving summary to %s" % summary_path)
    n.save(summary_path, summary)
    # job.show_summary_plots()

    # TODO either make work using suite3D function or remove?
    # if job.params['generate_sample_registered_bins']:
    #     sample_bin_path = os.path.join(job.dirs['summary'], 'sample_reg_movie.npy')
    #     sample_off_path = os.path.join(job.dirs['summary'], 'sample_offsets.npy')
    #     job.log("Registering sample files and saving them to %s for verification" % sample_bin_path)
    #     job.log("Offsets will be saved in summary.npy")

    #     init_mov, all_offsets = register_sample_movie(init_mov, all_ops, all_refs_masks, log_cb=job.log)

    #     n.save(sample_bin_path, init_mov)
    #     summary['all_offsets'] : all_offsets
    #     n.save(summary_path, summary)

    job.log("Initial pass complete. See %s for details" % job.dirs["summary"])
    
    if structural: 
        job.structural_summary = summary
    else:
        job.summary = summary
