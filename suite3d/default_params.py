import numpy as n
import copy

def get_default_params():
    return copy.deepcopy(params)


params = {

    ### Mandatory parameters
    'fs' : 2.8,
    'tau': 1.3,
    # Planes to analyze. 0 is deepest, 30 is shallowest
    # (the ScanImage channel mappings are corrected)
    'planes': n.arange(0, 30),
    # If you have less than 30 planes or you don't want to correct the channel mappings, set to False
    'convert_plane_ids_to_channel_ids' : True,
    'n_ch_tif' : 30, # number of channels in the recording


    ### File I/O ### 
    # Notch filter to remove line noise.
    # Should be a dictionary like:  {'f0' : 200, 'Q' : 1}
    # Where f0 is the frequency of the line noise, and Q is the quality factor
    'notch_filt' : None,

    ### Initialization Step ### 

    # number of files to use for the initialization step
    # Usually the equivalent of ~1 minute is enough
    'n_init_files' : 1,
    # list of tifs that are 'eligible' to be picked during initalization
    'init_file_pool' : None,
    # 'even' or 'random' sampling of init files
    'init_file_sample_method' : 'even',
    # number of random frames to select from initial files, set None for all
    'init_n_frames' : None,
    # make sure the mean image is all positive (add the offsets)
    'enforce_positivity' : True,

    # Crosstalk subtraction from pairs of planes 15 apart

    # To disable crosstalk subtraction, set this to False
    'subtract_crosstalk' : True,
    # if None, try to estimate crosstalk. 
    # if set to a float, it uses this as the crosstalk coefficient
    'override_crosstalk' : None,
    # Percentile: only consider pixels above this percentile when 
    # fitting the crosstalk coefficient
    'crosstalk_percentile' : 99.5, 
    # "smoothing" when estimating the crosstalk coefficient 
    'crosstalk_sigma' : 0.01,
    # number of planes (starting from top) used to estimate crosstalk
    # shallower (quiet) planes, especially those outside the brain,
    # lead to better estimates of the crosstalk
    'crosstalk_n_planes' : 10,

    ### Registration ###

    # parameters from suite2p
    'nonrigid' : True,
    'smooth_sigma' : 1.15,
    'maxregshift' : 0.15,
    'reg_filter_pcorr' : 1,

    # At the end of initalization, register and save an example bin
    # Could be useful to check registration parameters
    'generate_sample_registered_bins' : False,

    # Number of tifs to analyze at each batch. 
    # Larger batches require more memory, doesn't speed things up, so just leave it
    'tif_batch_size' : 1,

    ### Fusing ###
    # number of pixels to skip when stitching two strips together
    'n_skip' : 13,
    'fuse_crop' : None,
    # split the tif into smaller npy files of this length while fusing strips together
    'split_tif_size' : 100,


    ### SVD Decomposition ### 
    
    # 3-Tuple of 2-tuples of integers, defining the cropping bounds of the movie before
    # doing denoising, detection and everything else. Remove planes outside the brain,
    # and remove edges that are corrupted from inter-plane shifts
    # Example: svd_crop : ((0,14, (20,800), (50, 400)))
    'svd_crop' : None,
    # similarly, tuple of time indices to crop the movie to before following steps
    'svd_time_crop' : (None, None),

    # number of SVD components to compute per block
    'n_svd_comp' : 600,
    # Size of a block in z,y,x for block-based svd denoising
    'svd_block_shape' : (4,200, 200),
    # Overlap between blocks to remove artifacts at the edges of blocks
    # You don't have to do the math and make the block shape and overlaps
    # align with the movie size, it's done automatically
    # (0,0,0,) for no overlap
    'svd_block_overlaps' : (1,50,50),

    # internal parameter for Dask. how many pixels per dask "chunk"
    # best to leave at None, which defaults to the number of pixels in a block
    'svd_pix_chunk' : None,
    'svd_time_chunk' : 4000,
    'svd_save_time_chunk' : 400,
    'svd_save_comp_chunk' : 100,
    # When running the svd decomposition, how many blocks should be computed simultaneously
    # by dask. Too high leads to memory bottlenecks
    # Limited performance improvement by increasing this
    'n_svd_blocks_per_batch' : 1,

    ### Correlation Map ###

    # number of svd components to use in reconstruction is n_svd_comp
    # strength of normalization, 1.0 is standard. reduce below 1.0 (to ~0.8) if you see bright
    # blood vessels etc. in the correlation map
    'sdnorm_exp' : 1.0,

    # Type (gaussian, uniform) and xy/z extents of neuropil filter
    'npil_filt_type' : 'gaussian',
    'npil_filt_xy' : 5.0,
    'npil_filt_z' : 1.0,
    # Type and xy/z extents of the cell detection filter
    'conv_filt_type' : 'gaussian',
    'conv_filt_xy' : 1.0,
    'conv_filt_z' : 0.75,
    # activity threshold before calculating correlation map
    'intensity_thresh' : 0.25,
    # Width of the temporal hpf
    # Should divide t_batch_size evenly
    'temporal_hpf' : 400,
    
    # number of time points to process at each iteration
    # should be a multiple of temporal_hpf
    't_batch_size' : 200,
    # less important batchsize parameter for internal computations
    # for efficiency, should be t_batch_size / n_proc_corr
    'mproc_batchsize' : 25,
    # number of processors to use during correlation map calculation
    'n_proc_corr': 8,
    # don't touch this
    'dtype': n.float32,

    ### Cell detection ###
    
    # Size and overlap of cell detection patches
    'patch_size_xy' : (120,120),
    'patch_overlap_xy' : (25,25),
    # only consider timepoints with values above this threshold for detection
    'activity_thresh' : 4.0,
    # only consider timepoints above this percentile for detection. minimum thresh
    # between this and activity_thresh is used
    'percentile' : 99.0,
    # threshold to include a cell in an ROI. Lower to have larger ROIs
    'extend_thresh' : 0.2,

    # less useful parameters for cell detection:
    # number of extension iterations for each ROI. Recommend leaving at 2
    'roi_ext_iterations' : 2,
    # maximum number of ROIs that can be found in a patch
    'max_iter' : 10000 ,
    # Time binning factor for detection
    # if you have many samples per transient, consider increasing
    'detection_timebin' : 1 ,
    # Crop the movie before detection to only detect on a subset of the movie
    'detection_time_crop' : (None,None),
    # Allow overlap of cells (not functioning, can lead to weirdness)
    'allow_overlap' : False,
    # does nothing
    'recompute_v' : None,
    # normalize intensity of vmap planes before detection, not recommended
    'normalize_vmap' : False,
    # maximum number of pixels in a cell
    'max_pix' : 500,


    # Deconvolution
    # coefficient to multiply neuropil activity by before subtracting from cell activity
    'npil_coeff' : 0.7,
    # S2P deconvolution parameters
    'dcnv_baseline' : 'maximin',
    'dcnv_win_baseline' : 60,
    'dcnv_sig_baseline' : 10,
    'dcnv_prctile_baseline' : 8,
    'dcnv_batchsize' : 3000,
    # 'tau' : 1.3,

    # Legacy
    'subjects_dir' : None,
    'subject' : None,
    'expnum' : None,
    'date' : None,

}

