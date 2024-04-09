import os
import numpy as n
import copy

import matplotlib.pyplot as plt
from suite2p.registration import register
from . import utils
from . import lbmio
from . import reference_image as ref

def default_log(string, val): 
    print(string)

def choose_init_tifs(tifs, n_init_files, init_file_pool_lims=None, method='even', seed=2358):

    init_file_pool = []
    if init_file_pool_lims is not None:
        for limits in init_file_pool_lims:
            init_file_pool += tifs[limits[0]:limits[1]]
    else:
        init_file_pool = tifs
    init_file_pool = n.array(init_file_pool)
    
    if method == 'even':
        sample_file_ids = n.linspace(0, len(init_file_pool), n_init_files + 2, dtype=int)[1:-1]
        sample_tifs = n.array(init_file_pool)[sample_file_ids]
    elif method == 'random':
        # n.random.seed(seed)
        sample_tifs = n.random.choice(init_file_pool, n_init_files, replace=False)

    return sample_tifs

def load_init_tifs(init_tifs, planes, filter_params, n_ch_tif = 30, convert_plane_ids_to_channel_ids=True,fix_fastZ=False,):
    full_mov = lbmio.load_and_stitch_tifs(init_tifs, planes = planes, convert_plane_ids_to_channel_ids=convert_plane_ids_to_channel_ids,
                                          n_ch = n_ch_tif, filt=filter_params, concat=False, fix_fastZ=fix_fastZ)

    mov_lens = [mov.shape[1] for mov in full_mov] #not needed anymore?
    full_mov = n.concatenate(full_mov, axis=1)

    return full_mov

#TODO delete outdated if new reference function are used instead
def prep_registration(full_mov, reg_ops, log_callback=default_log, filter_pcorr=0, force_plane_shifts=None):
    nz, nt, ny, nx = full_mov.shape
    ref_img_3d = []
    log_callback("Computing reference images")
    for i in range(nz):
        ref_img = register.compute_reference(reg_ops, full_mov[i])
        ref_img_3d.append(ref_img)
        log_callback("  Computed reference for plane %d" % i,2)
    ref_img_3d = n.array(ref_img_3d)

    ref_img_3d_unaligned = n.copy(ref_img_3d)
    if force_plane_shifts is None:
        tvecs = n.concatenate([[[0,0]], utils.get_shifts_3d(ref_img_3d, filter_pcorr=filter_pcorr)])
    else: tvecs = force_plane_shifts
    log_callback("Tvecs: %s" % str(tvecs), 5)

    ref_img_3d_aligned = utils.register_movie(ref_img_3d[:,n.newaxis], tvecs=tvecs)[:,0]

    all_ref_and_masks = []
    all_ops = []
    for plane_idx in range(nz):
        ref_img = ref_img_3d_aligned[plane_idx].copy()
        plane_ops = copy.deepcopy(reg_ops)
        if plane_ops.get('norm_frames',False):
            plane_ops['rmin'], plane_ops['rmax'] = n.int16(n.percentile(ref_img,1)), n.int16(n.percentile(ref_img,99))
            ref_img = n.clip(ref_img, plane_ops['rmin'], plane_ops['rmax'])    
        ref_and_masks = register.compute_reference_masks(ref_img, plane_ops)
        all_ref_and_masks.append(ref_and_masks)
        all_ops.append(plane_ops)

    return tvecs, ref_img_3d_aligned, all_ops, all_ref_and_masks, ref_img_3d_unaligned

#Is this the old version, which uses suite 2p? 
def register_sample_movie(full_mov, all_ops, all_refs, in_place=True, log_callback=default_log):
    nz = full_mov.shape[0]
    if not in_place:
        full_mov = full_mov.copy()
    log_callback("Registering sample movie")
    all_offsets = []
    for plane_idx in range(nz):
        ref_and_masks = all_refs[plane_idx]
        plane_ops = all_ops[plane_idx]
        log_callback("  Registering plane %d" % plane_idx)
        full_mov[plane_idx], ym, xm, cm, ym1, xm1, cm1 = \
            register.register_frames(ref_and_masks, full_mov[plane_idx], ops=plane_ops)
        all_offsets.append((ym, xm, cm, ym1, xm1, cm1))

    return full_mov, all_offsets


def run_init_pass(job):
    tifs = job.tifs
    params = job.params

    summary_path = os.path.join(job.dirs['summary'], 'summary.npy')
    job.log("Saving summary to %s" % summary_path,0)
    if not os.path.isdir(job.dirs['summary']):
        job.log("Summary dir does not exist!!")
        assert False

    init_tifs = choose_init_tifs(tifs, params['n_init_files'], params['init_file_pool'], 
                                       params['init_file_sample_method'])
    n_ch_tif = job.params.get('n_ch_tif', 30)
    job.log("Loading init tifs with %d channels" % n_ch_tif)
    init_mov = load_init_tifs(
        init_tifs, params['planes'], params['notch_filt'], 
        n_ch_tif = n_ch_tif, fix_fastZ=params.get('fix_fastZ', False),
        convert_plane_ids_to_channel_ids = params.get('convert_plane_ids_to_channel_ids', True))
    job.log("Loaded init tifs")
    nz, nt, ny, nx = init_mov.shape
    if params['init_n_frames'] is not None:
        assert nt > params['init_n_frames'], 'not enough frames in loaded tifs'
        subset_ts = n.random.choice(n.arange(nt), params['init_n_frames'],replace=False)
        job.log("Selecting %d random frames from the init tif files" % params['init_n_frames'])
        init_mov = init_mov[:,subset_ts]
    nz, nt, ny, nx = init_mov.shape
    job.log("Loaded movie with %d frames and shape %d, %d, %d" % (nt, nz, ny, nx))
    im3d = init_mov.mean(axis=1)
    im3d_raw = im3d.copy()
    if job.params.get('enforce_positivity', False):
        # min_pix_vals = init_mov.min(axis=(1, 2, 3), keepdims=True)[:,0].astype(int)
        # min_pix_vals = n.percentile(init_mov.reshape(nz, -1), 1, axis=1).astype(int)
        min_pix_vals = im3d.min(axis=(1,2), keepdims=False).astype(int)
        job.log("Enforcing positivity in mean image",3)
        init_mov -= min_pix_vals[:, n.newaxis, n.newaxis, n.newaxis]
        
        # job.log("Min pix vals: %s" % str(min_pix_vals.flatten()), 3)
    else: min_pix_vals = None
    if params['subtract_crosstalk']:
        if params['override_crosstalk'] is not None:
            cross_coeff = params['override_crosstalk']
            job.log("Subtracting crosstalk with forced coefficient %0.3f" % cross_coeff)
        else:
            __, __, cross_coeff = utils.calculate_crosstalk_coeff(im3d,
                                                    estimate_from_last_n_planes=params['crosstalk_n_planes'],
                                                    sigma=params['crosstalk_sigma'], fit_above_percentile = params['crosstalk_percentile'],
                                                    show_plots=True, save_plots=job.dirs['summary'],
                                                    verbose=(job.verbosity == 2))
            job.log("Subtracting with estimated coefficient %0.3f" % cross_coeff)
            if cross_coeff > 0.4 or cross_coeff < 0.01:
                job.log("WARNING - seems like coefficient esimation failed!")
        for plane in params['planes']:
            if plane > 14 and plane - 15 in params['planes']:
                plane_idx = n.where(n.array(params['planes']) == plane)[0][0]
                sub_plane_idx = n.where(n.array(params['planes']) == plane - 15)[0][0]
                
                job.log("Subtracting plane %d from %d" % (plane-15, plane), 3)
                # job.log("        Corresponds to index %d from %d" % (sub_plane_idx, plane_idx))
                init_mov[plane_idx] = init_mov[plane_idx] - init_mov[sub_plane_idx] * cross_coeff
        im3d = init_mov.mean(axis=1)
    else:
        job.log("No crosstalk estimation or subtraction")
        cross_coeff = None

    if job.params.get('fuse_strips',True):
        job.log("Estimating fusing shifts")
        __, xs = lbmio.load_and_stitch_full_tif_mp(init_tifs[0], channels=n.arange(1), get_roi_start_pix=True, n_ch=n_ch_tif, fix_fastZ=job.params.get('fix_fastZ',False))
        fuse_shifts, fuse_ccs = utils.get_fusing_shifts(im3d_raw, xs)
        fuse_shift = int(n.round(n.median(fuse_shifts)))
        if job.params.get('fuse_shift_override', None) is not None:
            fuse_shift = int(job.params['fuse_shift_override'])
            job.log("Overriding", 2)
        job.log("Using best fuse shift of %d" % fuse_shift)
    else:
        fuse_shift = 0
        fuse_shifts = None
        fuse_ccs = None
    job.log("Building ops file")
    # return


    summary_mov_path = os.path.join(job.dirs['summary'], 'init_mov.npy')
    n.save(summary_mov_path, init_mov)
    job.log("Saved init mov to %s" % summary_mov_path)

    #TODO delta old method of calling function now changed
    # job.log("Aligning planes")
    # print(im3d.dtype)
    # if params.get('overwrite_plane_shifts') is not None:
    #     tvecs = params['overwrite_plane_shifts']
    # else:
    #     tvecs = n.concatenate([[[0,0]], utils.get_shifts_3d(im3d.astype(int), filter_pcorr = params['reg_filter_pcorr'],
    #                                                     n_procs = params['n_proc_corr'])])
    
    # if params.get('fix_shallow_plane_shift_estimates', True):
    #     shallow_plane_thresh = params.get('fix_shallow_plane_shift_esimate_threshold', 20)
    #     peaks = n.abs(tvecs[:shallow_plane_thresh]).max(axis=0)
    #     # print(peaks)
    #     # print(tvecs)peaks = n.abs(tvecs[:shallow_plane_thresh]).max(axis=0)
    #     bad_planes = n.logical_or(n.abs(tvecs[shallow_plane_thresh:,0]) > peaks[0], n.abs(tvecs[shallow_plane_thresh:,1]) > peaks[1])
    #     tvecs[shallow_plane_thresh:][bad_planes,:] =0
    #     job.log("Fixing %d plane alignment outliers" % bad_planes.sum(), 2)
    #     # print(tvecs)
    # job.log("Fusing and padding init mov")
    # if job.params.get('fuse_strips',True):
    #     init_mov, xpad, ypad, new_xs, og_xs = utils.pad_and_fuse(init_mov, plane_shifts=tvecs, fuse_shift=fuse_shift, xs=xs)
    # else:
    #     xpad = None; ypad = None; new_xs = None; og_xs = None
    # img_pad = init_mov.mean(axis=1)
    # __, __, ny, nx = init_mov.shape

    # reg_ops = utils.build_ops('', {}, {'smooth_sigma' : job.params['smooth_sigma'],
    #                                     'maxregshift' : job.params['maxregshift'],
    #                                     'Ly' : ny, 'Lx' : nx,
    #                                     'nonrigid' : job.params['nonrigid']})
    # tvecs, ref_img_3d, all_ops, all_refs_masks, ref_img_3d_unaligned = prep_registration(init_mov, reg_ops, job.log, force_plane_shifts=tvecs)
    # # ref_img_3d, ref_img_3d_unaligned, all_refs_masks, all_ops = None, None, None, None
    # summary = {
    #     'ref_img_3d' : ref_img_3d, # ctalk-sub and padded and plane-shifted
    #     'ref_img_3d_unaligned' : ref_img_3d_unaligned, #ctalk-sub and padded
    #     'raw_img' : im3d_raw, # right from the tiff file
    #     'img' : im3d, # crosstalk-subtracted
    #     'img_pad' : img_pad, #ctalk-sub and padded
    #     'crosstalk_coeff' : cross_coeff,
    #     'plane_shifts' : tvecs,
    #     'refs_and_masks' : all_refs_masks,
    #     'all_ops' : all_ops,
    #     'min_pix_vals' : min_pix_vals,
    #     'fuse_shifts' : fuse_shifts,
    #     'fuse_shift' : fuse_shift,
    #     'fuse_ccs' : fuse_ccs,
    #     'tiffile_xs' : xs,
    #     'xpad': xpad,
    #     'ypad' : ypad,
    #     'new_xs' : new_xs,
    #     'og_xs' : og_xs
    # }

    ##### new function
    reference_params = {'percent_contribute' : params.get('percent_contribute', 0.9), 
                        'block_size' : params.get('block_size', [128, 128]),
                        'Sigma' : params.get('sigma_reference', [1.45, 0]), #Y/X smooth, Z smooth
                        'smooth_sigma' : params.get('smooth_sigma_reference', 1.15), #spatial taper width
                        'niter' : params.get('n_reference_iterations', 8),
                        'max_reg_xy_reference' : params.get('max_reg_xy_reference', 50),
                        'batch_size' : params.get('gpu_reference_batch_size', 20), #keep in gpu RAM
                        }
    mov_fuse, new_xs, og_xs = ref.fuse_mov(init_mov, fuse_shift, xs)

    tvecs, ref_image, ref_padded, all_refs_and_masks, pad_sizes, reference_params = \
    ref.compute_reference_and_masks(mov_fuse, reference_params, log_callback = job.log , use_GPU = params.get('use_GPU_registration', True))

    summary = {
        'ref_img_3d' : ref_image, # ctalk-sub and padded and plane-shifted
        'ref_img_3d_unaligned' : ref_padded, #ctalk-sub and padded
        'raw_img' : im3d_raw, # right from the tiff file
        'img' : im3d, # crosstalk-subtracted
        'crosstalk_coeff' : cross_coeff,
        'plane_shifts' : tvecs[0],
        'plane_shifts_uncorrected' : tvecs[1],
        'refs_and_masks' : all_refs_and_masks,
        'reference_params' : reference_params,
        'min_pix_vals' : min_pix_vals,
        'fuse_shifts' : fuse_shifts,
        'fuse_shift' : fuse_shift,
        'fuse_ccs' : fuse_ccs,
        'tiffile_xs' : xs,
        'xpad': pad_sizes[0],
        'ypad' : pad_sizes[1],
        'new_xs' : new_xs,
        'og_xs' : og_xs
    }
    summary_path = os.path.join(job.dirs['summary'], 'summary.npy')
    job.log("Saving summary to %s" % summary_path)
    n.save(summary_path, summary)
    job.show_summary_plots()

    #TODO either make work using suite3D function or remove?
    # if job.params['generate_sample_registered_bins']:
    #     sample_bin_path = os.path.join(job.dirs['summary'], 'sample_reg_movie.npy')
    #     sample_off_path = os.path.join(job.dirs['summary'], 'sample_offsets.npy')
    #     job.log("Registering sample files and saving them to %s for verification" % sample_bin_path)
    #     job.log("Offsets will be saved in summary.npy")

    #     init_mov, all_offsets = register_sample_movie(init_mov, all_ops, all_refs_masks, log_callback=job.log)

    #     n.save(sample_bin_path, init_mov)
    #     summary['all_offsets'] : all_offsets
    #     n.save(summary_path, summary)


    job.log("Initial pass complete. See %s for details" % job.dirs['summary'])
    job.summary = summary

