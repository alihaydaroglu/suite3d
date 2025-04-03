import numpy as n
import copy
from .utils import set_num_processors


def get_default_params():
    return copy.deepcopy(params)


def get_matching_default_params(param_names):
    param_subset = {}
    for key in param_names:
        param_subset[key] = copy.copy(param_names[key])
    return param_subset


params = {
    ### Mandatory parameters
    "fs": 2.8,  # volume rate
    "tau": 1.3,  # GCamp6s
    "voxel_size_um": (15, 2.5, 2.5),  # size of a voxel in microns in z,y,x
    # Planes to analyze. 0 is deepest, 30 is shallowest
    # (the ScanImage channel mappings are corrected)
    "planes": n.arange(0, 30),
    # If you have less than 30 planes or you don't want to correct the channel mappings, set to False
    "convert_plane_ids_to_channel_ids": False,
    "n_ch_tif": 30,  # number of planes in the recording
    "skip_roi": None,  # if you want to skip one of the mROIs, enter its idx
    "lbm": True,  # whether the data is from light-bead microscopy
    "faced": False,  # whether data is from FACED microscopy,
    "faced_nz": None,  # number of z-planes in FACED data
    "multiplane_2p_use_planes": None, # Which planes to analyze in multiplane 2P data (this will only be applied if lbm=False and faced=False)
    # If True, to use safe mode for tif preregistration (slower but more reliable) -- only used if lbm=False and faced=False
    # In standard 2P data the number of planes in each tif might not divide evenly into the number of planes in the recording
    # so we need to preregister the tifs to figure out a loading strategy. 
    # The safe_mode loads all tifs to check their size, which is slow but reliable. safe_mode=False will guess the size in a way that we think
    # always works, so you should try this first and only use safe_mode=True if you run into problems. (Error messages are clear about this).
    "tif_preregistration_safe_mode": False, 
    ### File I/O ###
    # Notch filter to remove line noise.
    # Should be a dictionary like:  {'f0' : 200, 'Q' : 1}
    # Where f0 is the frequency of the line noise, and Q is the quality factor
    "notch_filt": None,
    "fix_fastZ": False,  # if you messed up your ROI z-definitions in scanimage, this is useful
    "num_colors": 1,  # if not lbm data, how many color channels were recorded by scanimage
    "functional_color_channel": 0,  # if not lbm data, which color channel is the functional one
    "save_dtype": "float16",
    ### Initialization Step ###
    # number of files to use for the initialization step
    # Usually the equivalent of ~1 minute is enough
    "n_init_files": 1,
    # list of tifs that are 'eligible' to be picked during initalization
    "init_file_pool": None,
    # 'even' or 'random' sampling of init files
    "init_file_sample_method": "even",
    # number of random frames to select from initial files, set None for all
    "init_n_frames": 500,
    # make sure the mean image is all positive (add the offsets)
    "enforce_positivity": True,
    # fix the plane shifts for top few planes that might be outside the brain
    "fix_shallow_plane_shift_estimates": False,
    "fix_shallow_plane_shift_esimate_threshold": 20,
    # 'overwrite_plane_shifts: set as a float array of size n_planes x 2 with (y,x) shifts for each plane
    "overwrite_plane_shifts": None,
    # Crosstalk subtraction from pairs of planes 15 apart
    # To disable crosstalk subtraction, set this to False
    "subtract_crosstalk": True,
    # if None, try to estimate crosstalk.
    # if set to a float, it uses this as the crosstalk coefficient
    "override_crosstalk": None,
    # Percentile: only consider pixels above this percentile when
    # fitting the crosstalk coefficient
    "crosstalk_percentile": 99.5,
    # "smoothing" when estimating the crosstalk coefficient
    "crosstalk_sigma": 0.01,
    # number of planes per cavity, so plane 0 will be subtracted from plane 0 + cavity_size
    "cavity_size": 15,
    # number of planes (starting from top) used to estimate crosstalk
    # shallower (quiet) planes, especially those outside the brain,
    # lead to better estimates of the crosstalk
    "crosstalk_n_planes": 2,
    ### Registration ###
    # whether or not to fuse the mesoscope strips
    "fuse_strips": True,
    # number of pixels to skip between strips - None will auto estimate
    "fuse_shift_override": None,
    # maximum rigid shift in pixels (includes plane-to-plane LBM shift, so make sure it's larger than that!)
    "max_rigid_shift_pix": 100,
    # whether or not to align each z plane in x/y
    "plane_to_plane_alignment": True,
    # number of frames per batch in gpu registration
    "gpu_reg_batchsize": 10,
    "max_shift_nr": 3,
    "nr_npad": 3,
    "nr_subpixel": 10,
    "nr_smooth_iters": 2,
    # 3d registration params
    "pc_size": n.asarray((2, 40, 40)),  # ~ max_reg_zyx
    "3d_reg": False,  # Use the new 3d registration fucntions
    "gpu_reg": False,
    # reference image paramaters
    "percent_contribute": 0.9,
    # percentage of frames which contribute to the reference image
    "block_size": (128, 128),  # if you have lots of non-rigid movement, consider (64, 64)
    # size of a non-rigid block
    "sigma_reference": (1.45, 0),
    "smooth_sigma_reference": 1.15,
    "n_reference_iterations": 8,
    "max_reg_xy_reference": 50,
    # max value in x/y which a plane can be shifted for the reference
    "gpu_reference_batch_size": 20,
    # parameters from suite2p
    "nonrigid": True,
    "apply_z_shift": False, 
    "smooth_sigma": 1.15,
    "maxregshift": 0.15,
    "reg_filter_pcorr": 1,
    "reg_norm_frames": True,  # clip frames during registration
    # At the end of initalization, register and save an example bin
    # Could be useful to check registration parameters
    "generate_sample_registered_bins": False,
    # Number of tifs to analyze at each batch.
    # Larger batches require more memory, doesn't speed things up, so just leave it
    "tif_batch_size": 1,
    ### Fusing (ONLY FOR STANDALONE FUSING - USE REGISTRATION PARAMS FOR FUSING DURING REGISTRATION!) ###
    # number of pixels to skip when stitching two strips together
    "n_skip": 13,
    "fuse_crop": None,
    # split the tif into smaller npy files of this length while fusing strips together
    "split_tif_size": 100,
    ### SVD Decomposition ###
    # 3-Tuple of 2-tuples of integers, defining the cropping bounds of the movie before
    # doing denoising, detection and everything else. Remove planes outside the brain,
    # and remove edges that are corrupted from inter-plane shifts
    # Example: svd_crop : ((0,14, (20,800), (50, 400)))
    "svd_crop": None,
    # similarly, tuple of time indices to crop the movie to before following steps
    "svd_time_crop": (None, None),
    # number of SVD components to compute per block
    "n_svd_comp": 600,
    # Size of a block in z,y,x for block-based svd denoising
    "svd_block_shape": (4, 200, 200),
    # Overlap between blocks to remove artifacts at the edges of blocks
    # You don't have to do the math and make the block shape and overlaps
    # align with the movie size, it's done automatically
    # (0,0,0,) for no overlap
    "svd_block_overlaps": (1, 50, 50),
    # internal parameter for Dask. how many pixels per dask "chunk"
    # best to leave at None, which defaults to the number of pixels in a block
    "svd_pix_chunk": None,
    "svd_time_chunk": 4000,
    "svd_save_time_chunk": 400,
    "svd_save_comp_chunk": 100,
    # When running the svd decomposition, how many blocks should be computed simultaneously
    # by dask. Too high leads to memory bottlenecks
    # Limited performance improvement by increasing this
    "n_svd_blocks_per_batch": 1,
    ### Correlation Map ###
    # number of svd components to use in reconstruction is n_svd_comp
    # strength of normalization, 1.0 is standard. reduce below 1.0 (to ~0.8) if you see bright
    # blood vessels etc. in the correlation map
    "sdnorm_exp": 1.0,
    # crop the edges of each plane by this many pixels before computing the corr map
    # this removes some registration-related artifacts
    "edge_crop_npix": 7,
    # Type (gaussian, unif) and xy/z extents of neuropil filter in pixels
    "npil_filt_type": "unif",
    "npil_filt_xy_um": 70.0,
    "npil_filt_z_um": 15.0,
    # Type and xy/z extents of the cell detection filter in pixels
    "cell_filt_type": "unif",
    "cell_filt_xy_um": 10,
    "cell_filt_z_um": 15,
    # activity threshold before calculating correlation map
    "intensity_thresh": 0.1,
    "standard_vmap": True,  # use suite2p algorithm for vmap
    # Width of the temporal hpf
    # Should divide t_batch_size evenly
    "temporal_hpf": 200,
    # sometimes, the top and bottom planes have different scales
    # than the center planes in the correlation map. Attempt to fix it
    "fix_vmap_edge_planes": False,
    # number of time points to process at each iteration
    # should be a multiple of temporal_hpf
    "t_batch_size": 200,
    # less important batchsize parameter for internal computations
    # for efficiency, should be t_batch_size / n_proc_corr
    "mproc_batchsize": 25,
    # number of processors to use during correlation map calculation
    "n_proc": set_num_processors(16),
    "n_proc_corr": set_num_processors(16),
    "n_proc_detect": set_num_processors(16),
    # don't touch this
    "dtype": n.float32,
    ### Cell segmentation ###
    # threshold above which cell peaks in correlation map are detected
    "peak_thresh": 2.0,
    # Size and overlap of cell segmentation patches
    "patch_size_xy": (120, 120),
    "patch_overlap_xy": (25, 25),
    # only consider timepoints with values above this threshold for segmentation
    "activity_thresh": 20.0,
    # only consider timepoints above this percentile for segmentation. minimum thresh
    # between this and activity_thresh is used
    "percentile": 99.5,
    # threshold to include a cell in an ROI. Lower to have larger ROIs
    "extend_thresh": 0.2,
    # less useful parameters for cell segmentation:
    # number of extension iterations for each ROI. Recommend leaving at 2
    "roi_ext_iterations": 2,
    # number of iterations around a cell to exclude future cells from
    "ext_subtract_iters": 0,
    # maximum number of ROIs that can be found in a patch
    "max_iter": 10000,
    # Time binning factor for segmentation
    # if you have many samples per transient, consider increasing
    "detection_timebin": 1,
    "segmentation_timebin": 1,
    # Crop the movie before segmentation to only detect on a subset of the movie
    "detection_time_crop": (None, None),
    # Allow overlap of cells (not functioning, can lead to weirdness)
    "allow_overlap": False,
    # does nothing
    "recompute_v": None,
    # normalize intensity of vmap planes before segmentation, not recommended
    "normalize_vmap": False,
    # maximum number of pixels in a cell
    "max_pix": 500,
    # remove duplicate cells that are closer than dist_thresh and share more weighted pixels than lam_thresh
    "detect_overlap_dist_thresh": 5,
    "detect_overlap_lam_thresh": 0.5,
    # Deconvolution
    # coefficient to multiply neuropil activity by before subtracting from cell activity
    "npil_coeff": 0.7,
    "npil_to_roi_npix_ratio": None,
    "min_npil_npix": 100,
    # S2P deconvolution parameters
    "dcnv_baseline": "maximin",
    "dcnv_win_baseline": 60,
    "dcnv_sig_baseline": 10,
    "dcnv_prctile_baseline": 8,
    "dcnv_batchsize": 3000,
    # 'tau' : 1.3,
    # Legacy
    "subjects_dir": None,
    "subject": None,
    "expnum": None,
    "date": None,
}
