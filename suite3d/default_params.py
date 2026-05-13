import numpy as n
import copy
from .utils import set_num_processors


# =============================================================================
# Parameter sections
#
# Each parameter belongs to exactly one section. This mapping is used to:
# 1. Extract the relevant subset of params for each pipeline step
# 2. Validate that user-provided params are recognized
# 3. Document which params affect which step
#
# Sections:
#   data        - describes the data itself (acquisition, microscope, format)
#   init        - initialization pass (reference image, crosstalk, plane alignment)
#   registration - rigid/nonrigid registration of frames to reference
#   corrmap     - correlation map computation (filtering, neuropil subtraction)
#   segmentation - cell detection from correlation map
#   extraction  - trace extraction and deconvolution
#   compute     - computational settings (processors, dtypes, batch sizes)
# =============================================================================

PARAM_SECTIONS = {
    # --- data: describes the acquisition and data format ---
    "fs":                           "data",
    "tau":                          "data",
    "voxel_size_um":                "data",
    "planes":                       "data",
    "convert_plane_ids_to_channel_ids": "data",
    "n_ch_tif":                     "data",
    "skip_roi":                     "data",
    "lbm":                          "data",
    "faced":                        "data",
    "faced_nz":                     "data",
    "multiplane_2p_use_planes":     "data",
    "tif_preregistration_safe_mode": "data",
    "notch_filt":                   "data",
    "fix_fastZ":                    "data",
    "num_colors":                   "data",
    "functional_color_channel":     "data",
    "process_structural_channel":   "data",
    "structural_color_channel":     "data",
    "clear_registered_structural_data": "data",

    # --- init: initialization pass ---
    "n_init_files":                 "init",
    "init_file_pool":               "init",
    "init_file_sample_method":      "init",
    "init_n_frames":                "init",
    "enforce_positivity":           "init",
    "fix_shallow_plane_shift_estimates":   "init",
    "fix_shallow_plane_shift_estimate_threshold": "init",
    "overwrite_plane_shifts":       "init",
    "subtract_crosstalk":           "init",
    "override_crosstalk":           "init",
    "crosstalk_percentile":         "init",
    "crosstalk_sigma":              "init",
    "cavity_size":                  "init",
    "crosstalk_n_planes":           "init",
    "fuse_strips":                  "init",
    "fuse_shift_override":          "init",
    "plane_to_plane_alignment":     "init",

    # --- registration: rigid and nonrigid registration ---
    "max_rigid_shift_pix":          "registration",
    "gpu_reg_batchsize":            "registration",
    "max_shift_nr":                 "registration",
    "nr_npad":                      "registration",
    "nr_subpixel":                  "registration",
    "nr_smooth_iters":              "registration",
    "save_nonrigid_phasecorrs":     "registration",
    "pc_size":                      "registration",
    "3d_reg":                       "registration",
    "gpu_reg":                      "registration",
    "percent_contribute":           "registration",
    "block_size":                   "registration",
    "sigma_reference":              "registration",
    "smooth_sigma_reference":       "registration",
    "n_reference_iterations":       "registration",
    "max_reg_xy_reference":         "registration",
    "gpu_reference_batch_size":     "registration",
    "block_size_3d":                "registration",
    "nonrigid":                     "registration",
    "apply_z_shift":                "registration",
    "smooth_sigma_nr":              "registration",
    "smooth_sigma":                 "registration",
    "snr_thresh":                   "registration",
    "maxregshift":                  "registration",
    "smooth_sigma_time":            "registration",
    "reg_filter_pcorr":             "registration",
    "reg_norm_frames":              "registration",
    "tif_batch_size":               "registration",
    "n_skip":                       "registration",
    "fuse_crop":                    "registration",
    "split_tif_size":               "registration",
    "generate_sample_registered_bins": "registration",

    # --- corrmap: correlation map computation ---
    "svd_crop":                     "corrmap",
    "svd_time_crop":                "corrmap",
    "n_svd_comp":                   "corrmap",
    "svd_block_shape":              "corrmap",
    "svd_block_overlaps":           "corrmap",
    "svd_pix_chunk":                "corrmap",
    "svd_time_chunk":               "corrmap",
    "svd_save_time_chunk":          "corrmap",
    "svd_save_comp_chunk":          "corrmap",
    "n_svd_blocks_per_batch":       "corrmap",
    "sdnorm_exp":                   "corrmap",
    "edge_crop_npix":               "corrmap",
    "npil_filt_type":               "corrmap",
    "npil_filt_xy_um":              "corrmap",
    "npil_filt_z_um":               "corrmap",
    "cell_filt_type":               "corrmap",
    "cell_filt_xy_um":              "corrmap",
    "cell_filt_z_um":               "corrmap",
    "intensity_thresh":             "corrmap",
    "standard_vmap":                "corrmap",
    "temporal_hpf":                 "corrmap",
    "fix_vmap_edge_planes":         "corrmap",
    "t_batch_size":                 "corrmap",
    "detection_timebin":            "corrmap",

    # --- segmentation: cell detection from correlation map ---
    "peak_thresh":                  "segmentation",
    "extend_func":                  "segmentation",
    "patch_size_xy":                "segmentation",
    "patch_overlap_xy":             "segmentation",
    "activity_thresh":              "segmentation",
    "percentile":                   "segmentation",
    "vox_snr_thresh":               "segmentation",
    "multi_source":                 "segmentation",
    "n_power_iter":                 "segmentation",
    "use_power_iter_v1":            "segmentation",
    "min_frames":                   "segmentation",
    "roi_ext_iterations":           "segmentation",
    "roi_dilations_per_iter":       "segmentation",
    "ext_subtract_iters":           "segmentation",
    "vox_snr_mp_correction":        "segmentation",
    "max_iter":                     "segmentation",
    "segmentation_timebin":         "segmentation",
    "segmentation_spatial_filt":    "segmentation",
    "detection_time_crop":          "segmentation",
    "local_thresh":                 "segmentation",
    "local_thresh_window_pix":      "segmentation",
    "local_thresh_pct":             "segmentation",
    "allow_overlap":                "segmentation",
    "max_pix":                      "segmentation",
    "detect_overlap_dist_thresh":   "segmentation",
    "detect_overlap_lam_thresh":    "segmentation",

    # --- extraction: neuropil subtraction, trace extraction, deconvolution ---
    "npil_coeff":                   "extraction",
    "npil_to_roi_npix_ratio":       "extraction",
    "min_npil_npix":                "extraction",
    "dcnv_baseline":                "extraction",
    "dcnv_win_baseline":            "extraction",
    "dcnv_sig_baseline":            "extraction",
    "dcnv_prctile_baseline":        "extraction",
    "dcnv_batchsize":               "extraction",
    "deduplicate":                  "extraction",
    "deduplication_thresh_um":      "extraction",
    "deduplication_thresh_corr":    "extraction",

    # --- compute: computational settings ---
    "n_proc":                       "compute",
    "n_proc_corr":                  "compute",
    "n_proc_detect":                "compute",
    "dtype":                        "compute",
    "save_dtype":                   "compute",
}


def get_default_params():
    """Return a deep copy of the default parameter dictionary."""
    return copy.deepcopy(params)


def get_section_params(params_dict, *sections):
    """Extract parameters belonging to one or more sections.

    Args:
        params_dict (dict): Full parameter dictionary.
        *sections (str): One or more section names (e.g., 'corrmap', 'segmentation').

    Returns:
        dict: Subset of params_dict containing only keys in the given sections.
    """
    return {
        k: v for k, v in params_dict.items()
        if PARAM_SECTIONS.get(k) in sections
    }


def get_known_param_names():
    """Return the set of all recognized parameter names."""
    return set(PARAM_SECTIONS.keys())


def validate_params(user_params):
    """Check that all user-provided parameter names are recognized.

    Args:
        user_params (dict): User-provided parameters to validate.

    Raises:
        ValueError: If any parameter name is not recognized.
    """
    known = get_known_param_names()
    # Allow internal keys like 'tifs', 'frame_counts' etc.
    internal_keys = {"tifs", "frame_counts", "extra_frames", "previous_tif"}
    unknown = set(user_params.keys()) - known - internal_keys
    if unknown:
        raise ValueError(
            "Unknown parameter(s): %s. "
            "Valid parameters: %s" % (sorted(unknown), sorted(known))
        )


# =============================================================================
# Default parameters
#
# Organized by section. Comments serve as documentation for each parameter.
# =============================================================================

params = {
    # =========================================================================
    # DATA: describes the acquisition and data format
    # =========================================================================
    "fs": 2.8,                          # volume rate (Hz)
    "tau": 1.3,                         # GCaMP decay time (s). 1.3 for GCaMP6s
    "voxel_size_um": (15, 2.5, 2.5),    # voxel size in microns (z, y, x)
    "planes": n.arange(0, 30),          # planes to analyze (0 = deepest)
    "convert_plane_ids_to_channel_ids": False,
    "n_ch_tif": 30,                     # number of planes per volume in TIFF
    "skip_roi": None,                   # skip this mROI index (None = don't skip)
    "lbm": True,                        # True for Light Beads Microscopy data
    "faced": False,                     # True for FACED microscopy data
    "faced_nz": None,                   # number of z-planes in FACED data
    "multiplane_2p_use_planes": None,   # which planes for standard multiplane 2P
    "tif_preregistration_safe_mode": False,  # slower but more reliable tif sizing
    "notch_filt": None,                 # line noise removal: {'f0': freq, 'Q': quality}
    "fix_fastZ": False,                 # fix ROI z-definition errors in ScanImage
    "num_colors": 1,                    # color channels recorded by ScanImage
    "functional_color_channel": 0,      # which channel is functional
    "process_structural_channel": False, # process a structural color channel
    "structural_color_channel": 1,      # which channel is structural
    "clear_registered_structural_data": True,

    # =========================================================================
    # INIT: initialization pass (reference image, crosstalk, plane alignment)
    # =========================================================================
    "n_init_files": 1,                  # number of TIFFs for initialization (~500 frames)
    "init_file_pool": None,             # restrict init files to this pool
    "init_file_sample_method": "even",  # 'even' or 'random' sampling
    "init_n_frames": 500,               # random frames from init files (None = all)
    "enforce_positivity": True,         # shift mean image to be all-positive
    "fix_shallow_plane_shift_estimates": False,
    "fix_shallow_plane_shift_estimate_threshold": 20,
    "overwrite_plane_shifts": None,     # manually set plane shifts (nz x 2 array)
    # Crosstalk subtraction (LBM)
    "subtract_crosstalk": True,
    "override_crosstalk": None,         # force this crosstalk coefficient (float)
    "crosstalk_percentile": 99.0,
    "crosstalk_sigma": 0.01,
    "cavity_size": 15,                  # planes per cavity
    "crosstalk_n_planes": 2,            # planes used to estimate crosstalk
    # Strip fusion (mesoscope)
    "fuse_strips": True,
    "fuse_shift_override": None,        # override automatic fuse shift (int)
    "plane_to_plane_alignment": True,   # align z-planes in x/y

    # =========================================================================
    # REGISTRATION: rigid and nonrigid frame registration
    # =========================================================================
    "max_rigid_shift_pix": 100,         # maximum rigid shift (pixels)
    "gpu_reg_batchsize": 10,            # frames per GPU batch
    "3d_reg": True,                     # use 3D registration
    "gpu_reg": True,                    # use GPU acceleration
    # Reference image
    "percent_contribute": 0.9,          # fraction of frames for reference
    "block_size": (128, 128),           # nonrigid block size (y, x)
    "sigma_reference": (1.45, 0),
    "smooth_sigma_reference": 1.15,
    "n_reference_iterations": 8,
    "max_reg_xy_reference": 50,         # max reference xy shift
    "gpu_reference_batch_size": 20,
    "block_size_3d": (5, 128, 128),     # nonrigid 3D block (z, y, x)
    "pc_size": n.asarray((2, 40, 40)),  # phase correlation window
    # Nonrigid registration
    "nonrigid": False,
    "apply_z_shift": True,              # apply the rigid z component during shift application; set False to keep 3D measurement but skip z apply (e.g. few-z-plane recordings where z phase-corr saturates)
    "smooth_sigma_nr": 1.15,
    "smooth_sigma": 1.15,
    "snr_thresh": 1.2,                  # SNR threshold for nonrigid (2D)
    "maxregshift": 0.15,                # max shift fraction (2D CPU only)
    "smooth_sigma_time": 0,             # temporal smoothing (2D CPU only)
    "max_shift_nr": 3,
    "nr_npad": 3,
    "nr_subpixel": 10,
    "nr_smooth_iters": 2,
    "save_nonrigid_phasecorrs": False,
    "reg_filter_pcorr": 1,
    "reg_norm_frames": True,            # clip frames during registration
    "tif_batch_size": 1,                # TIFFs per batch
    # Fusing / file splitting
    "n_skip": 13,                       # pixels to skip between strips
    "fuse_crop": None,
    "split_tif_size": 100,              # split registered data into chunks

    # =========================================================================
    # CORRMAP: correlation map (filtering, neuropil subtraction)
    # =========================================================================
    # SVD decomposition
    "svd_crop": None,                   # crop before SVD: ((z0,z1), (y0,y1), (x0,x1))
    "svd_time_crop": (None, None),      # time crop before SVD
    "n_svd_comp": 600,                  # SVD components per block
    "svd_block_shape": (4, 200, 200),   # block size (z, y, x)
    "svd_block_overlaps": (1, 50, 50),  # overlap between blocks
    "svd_pix_chunk": None,              # dask chunk size (None = auto)
    "svd_time_chunk": 4000,
    "svd_save_time_chunk": 400,
    "svd_save_comp_chunk": 100,
    "n_svd_blocks_per_batch": 1,
    # Filtering
    "sdnorm_exp": 0.9,                  # normalization strength (lower = less bright vessels)
    "edge_crop_npix": 7,                # edge pixels to crop
    "npil_filt_type": "unif",           # neuropil filter type
    "npil_filt_xy_um": 100.0,           # neuropil filter xy extent (um)
    "npil_filt_z_um": 15.0,             # neuropil filter z extent (um)
    "cell_filt_type": "gaussian",       # cell detection filter type
    "cell_filt_xy_um": 1.5,              # cell filter xy extent (um)
    "cell_filt_z_um": 10,               # cell filter z extent (um)
    "intensity_thresh": 5,               # activity threshold for corrmap
    "standard_vmap": True,              # suite2p-inspired vmap algorithm
    "temporal_hpf": 200,                # temporal high-pass filter width
    "fix_vmap_edge_planes": False,      # fix edge plane scaling
    "t_batch_size": 800,                # frames per corrmap batch
    "detection_timebin": None,          # time binning (None = auto from fs/tau)

    # =========================================================================
    # SEGMENTATION: cell detection from correlation map
    # =========================================================================
    "peak_thresh": 0.1,                 # corrmap peak detection threshold
    "extend_func": "corr",              # 'corr' or 'proj' for ROI extension
    "patch_size_xy": (150, 150),        # segmentation patch size (y, x)
    "patch_overlap_xy": (25, 25),       # patch overlap
    "activity_thresh": 5.0,             # minimum activity for segmentation
    "percentile": 95.0,                 # activity percentile threshold
    "vox_snr_thresh": 0.05,             # voxel SNR threshold for ROI inclusion
    "multi_source": True,               # multi-source correction
    "n_power_iter": 3,                  # power iterations for footprints
    "use_power_iter_v1": True,          # use power iteration for v1
    "min_frames": 50,                   # minimum frames per patch
    "roi_ext_iterations": 20,            # ROI extension iterations
    "roi_dilations_per_iter": 2,
    "ext_subtract_iters": 0,            # exclusion iterations around cells
    "vox_snr_mp_correction": False,     # Marchenko-Pastur correction for voxel SNR during ROI extension
    "max_iter": 10000,                  # max ROIs per patch
    "segmentation_timebin": 1,          # time binning for segmentation
    "segmentation_spatial_filt": 2,     # uniform filter size in pixels (applied in xy per frame)
    "detection_time_crop": (None, None),
    # Local thresholding
    "local_thresh": True,
    "local_thresh_window_pix": 51,
    "local_thresh_pct": 50,
    # ROI constraints
    "allow_overlap": False,             # experimental, not fully functional
    "max_pix": 10000,                    # maximum pixels per cell
    "detect_overlap_dist_thresh": 5,    # duplicate detection distance
    "detect_overlap_lam_thresh": 0.5,   # duplicate detection overlap

    # =========================================================================
    # EXTRACTION: neuropil subtraction, trace extraction, deconvolution
    # =========================================================================
    "npil_coeff": 0.7,                  # neuropil subtraction coefficient
    "npil_to_roi_npix_ratio": None,
    "min_npil_npix": 100,               # minimum neuropil pixels
    # Deconvolution (OASIS via suite2p)
    "dcnv_baseline": "maximin",
    "dcnv_win_baseline": 60,
    "dcnv_sig_baseline": 10,
    "dcnv_prctile_baseline": 8,
    "dcnv_batchsize": 3000,
    # Post-extraction deduplication: merge nearby cells with highly correlated traces
    "deduplicate": False,                # enable post-extraction deduplication
    "deduplication_thresh_um": 15.0,     # max centroid distance (microns) to consider a pair
    "deduplication_thresh_corr": 0.95,   # min trace correlation to merge a pair

    # =========================================================================
    # COMPUTE: computational settings (processors, dtypes, batch sizes)
    # =========================================================================
    "n_proc": set_num_processors(16),
    "n_proc_corr": set_num_processors(16),
    "n_proc_detect": set_num_processors(16),
    "dtype": n.float32,
    "save_dtype": "float16",
}
