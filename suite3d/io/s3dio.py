import tifffile
import numpy as n
import time
from ..developer import todo, deprecated_inputs
from .lbmio import (
    load_and_stitch_full_tif_mp,
    convert_lbm_plane_to_channel,
    get_roi_start_pix,
)


class s3dio:
    """
    A class that handles all operations related to data loading and preprocessing for suite3d.

    This class makes data-loading and preprocessing easy. You just have to set the parameters
    of the job class that it takes as input, then this class will handle all the rest. The goal
    is to make the interface of data-loading as straightforward as possible, with minimal need
    to consider implementation details related to the particular requirements of the job.

    An instance of this class can be passed as an argument to other modules to make sure data
    loading is handled in a consistent and easy-to-use way wherever relevant.
    """

    def __init__(self, job):
        # By containing a reference to the job object, we can easily access all relevant
        # parameters related to data-loading without having to remember all the kwargs.
        self.job = job

    def _update_prms(self, **parameters):
        """
        A helper function that updates the job parameters with the provided parameters.
        """
        use_params = self.job.params.copy()  # get default parameters
        use_params.update(parameters)  # update with any provided in kwargs
        return use_params  # return the parameters intended for use right now

    def _lbm_plane_to_ch(self, lbm_plane_to_ch):
        """
        A helper function that maps the LBM plane IDs to the channel IDs.
        """
        return n.array(lbm_plane_to_ch) - 1

    def _get_dataloader(self, params):
        """
        A function that returns the appropriate loader function based on the job's parameters.
        """
        if params["lbm"]:
            return self._load_lbm_tifs
        elif params["faced"]:
            return self._load_faced_tifs
        else:
            return self._load_scanimage_tifs

    def _preprocess_data(self, mov, params):
        """
        A central mechanism for preprocessing data. This function is meant to be called every
        single time raw data files are loaded, and it will handle all the necessary steps to
        ensure that the raw data files are loaded correctly depending on the job's parameters (lbm or not),
        any "local" parameters (like the ones typically passed as kwargs to ``lbmio.load_and_stitch_tifs()``),
        and anything else that is needed.
        """
        # example use of _update_prms to get the parameters to use for this call
        todo("This function is not implemented yet. Please implement it before using it!")
        return mov

    def load_data(self, paths, verbose=True, debug=False, structural=False, **parameters):
        """
        A central mechanism for loading data files. This function is meant to be called every
        single time raw data files are ever loaded, and it will handle all the necessary steps to
        ensure that the raw data files are loaded correctly depending on the job's parameters (lbm or not),
        any "local" parameters (like the ones typically passed as kwargs to ``lbmio.load_and_stitch_tifs()``),
        and anything else that is needed.
        """
        # example use of _update_prms to get the parameters to use for this call
        params = self._update_prms(**parameters)
        _dataloader = self._get_dataloader(params)
        mov_list = _dataloader(paths, params, verbose=verbose, debug=debug, structural=structural)
        # concatenate movies across time to make a single movie
        mov = n.concatenate(mov_list, axis=1)

        if verbose:
            size = mov.nbytes / (1024**3)
            self.job.log(f"Loaded {len(paths)} files, total {size:.2f} GB", 1)

        return self._preprocess_data(mov, params)

    def _load_scanimage_tifs(self, paths, params, verbose=True, debug=False, structural=False):
        """
        Load tifs that are in the standard 2p imaging format from ScanImage.

        Typical 2P data with multiplanes and multiple channels is organized
        as a stack of tiffs as follows (where F stands for Frame and P stands
        for Plane, with N total planes and T total frames).

        F1 - P1 - green
        F1 - P1 - red
        ...
        F1 - PN - green
        F1 - PN - red
        ...
        F2 - P1 - green
        F2 - P1 - red
        ...
        FT - PN - green
        FT - PN - red

        Args:
            paths (list): list of absolute paths to tiff files
            params (dict): parameters for loading the tiffs (inherited from self.job.params in the caller)
            verbose (bool, optional): Verbosity. Defaults to True.
            debug (bool, optional): Debugging mode. Defaults to False.
            structural (bool, optional): Whether to process the structural channel. Defaults to False.
                                         If True, params["structural_color_channel"] will be used to load the structural channel.
        Returns:
            mov (ndarray): the loaded tiff data with shape (planes, frames, y-pixels, x-pixels)
        """
        def filter_color_channel(mov):
            if params["num_colors"] > 1:
                # TODO: make sure that tiffs are 3d when num_colors==1
                # get functional channel from multi-channel tiff
                
                if len(mov.shape) == 3:
                    mov = mov.reshape(-1, params['num_colors'], mov.shape[-2],mov.shape[-1])
                elif len(mov.shape) != 4:
                    raise ValueError(
                        f"tiff file is {mov.ndim}D instead of 4D, expecting (frames, colors, y-pixels, x-pixels)"
                    )
                if mov.shape[1] != params["num_colors"]:
                    raise ValueError(
                        f"tiffs have {mov.shape[1]} color channels, expecting {params['num_colors']}"
                    )

                # in general, imaging is only done with one functional color channel, so we take that one and ignore the others
                # if anyone is using multiple functional color channels, they need to modify the code themselves or raise an
                # issue to ask for this feature to be implemented. A simple work around is to run the suite3d pipeline multiple
                # times with different functional color channels and then combine the results however you see fit.
                color_channel = params["functional_color_channel"] if not structural else params["structural_color_channel"]
                mov = n.take(mov, color_channel, axis=1)
            return mov

        # todo("Should we filter across the slow y-axis like the lbm loader?")

        if any([p < 0 or p >= params["n_ch_tif"] for p in params["planes"]]):
            raise ValueError(
                f"Planes must be in range 0-{params['n_ch_tif']}, but it's set to: {params['planes']}"
            )

        # Check preregistration
        required_preregistration_params = ["frame_counts", "extra_frames", "previous_tif"]
        for param in required_preregistration_params:
            if param not in params:
                raise ValueError(f"{param} not found in params -- preregister_tifs() must be called before loading tiffs!")

        tic = time.time()

        mov_list = []
        mov_extra_frames = {}
        for itif, tif_path in enumerate(paths):
            if verbose:
                self.job.log(f"Loading tiff {itif+1}/{len(paths)}: {tif_path}", 2)

            if tif_path not in params["frame_counts"] or tif_path not in params["extra_frames"] or tif_path not in params["previous_tif"]:
                raise ValueError(f"tif_path {tif_path} not found in frame_counts, extra_frames, or previous_tif - did the list of tifs change somehow?")
            
            # Load current tif file
            tif_file = filter_color_channel(tifffile.imread(tif_path))
            
            # Check if frames match expected number of frames
            expected_frames = params["frame_counts"][tif_path]
            if tif_file.shape[0] != expected_frames:
                raise ValueError(
                    f"tif_path {tif_path} has {tif_file.shape[0]} frames, but preregister_tifs() detected {expected_frames} frames."
                    "This may be caused by using preregister_tifs() in safe_mode=False which is fast but error prone." 
                    "Please set tif_preregistration_safe_mode=True in your params and try again."
                    "If that still doesn't work, then either the files have changed or there is inconcsistency in the tif structure!"
                )

            # Get the number of frames in previous tifs
            c_prev_tif = params["previous_tif"][tif_path]
            frames_from_previous = params["extra_frames"][c_prev_tif] if c_prev_tif else 0

            if frames_from_previous > 0:
                # We need to load the previous tif file and add it's extra frames
                if c_prev_tif in mov_extra_frames:
                    # This means we already loaded it and collected the extra frames
                    frames_from_previous = mov_extra_frames[c_prev_tif]
                else:
                    # We need to load the previous tif file and collect the extra frames
                    prev_tif_file = tifffile.imread(c_prev_tif)
                    frames_from_previous = filter_color_channel(prev_tif_file[-frames_from_previous:])

                tif_file = n.concatenate([frames_from_previous, tif_file], axis=0)

            # Now that we've added previous frames if necessary, we can remove final frames
            n_frames_total = tif_file.shape[0]
            check_extra_frames = n_frames_total % params["n_ch_tif"]
            extra_frames_expected = params["extra_frames"][tif_path]
            if check_extra_frames != extra_frames_expected:
                raise ValueError(f"tif_path {tif_path} has {check_extra_frames} extra frames, but preregister_tifs() detected {extra_frames_expected} frames.")
            
            # Remove extra frames and cache them for later if needed
            if extra_frames_expected > 0:   
                extra_frames = tif_file[-extra_frames_expected:]
                mov_extra_frames[tif_path] = extra_frames

                # Only keep a full volume of frames
                tif_file = tif_file[:-extra_frames_expected]

            if tif_file.ndim != 3:
                raise ValueError(
                    f"tiff file is not 3D, expecting (num_frames_not_volume!, y-pixels, x-pixels), but it has shape {tif_file.shape}"
                )
            
            if debug:
                print(f":Loading time up to tiff #{itif+1}: {time.time() - tic:.4f} s")

            # Do a final check that the tif matches volumes
            t, py, px = tif_file.shape
            volumes = t // params["n_ch_tif"]
            if volumes * params["n_ch_tif"] != t:
                raise ValueError(
                    f"tif_path {tif_path} has {t} frames which still doesn't divide evenly into volumes ({t%params['n_ch_tif']} frames left over)."
                    "This may be caused by using preregister_tifs() in safe_mode=False which is fast but error prone."  
                    "Please set tif_preregistration_safe_mode=True in your params and try again."
                    "If that still doesn't work, then either the files have changed or there is inconcsistency in the tif structure!"
                )

            # Reshape the movie to have dimensions (planes, frames, y-pixels, x-pixels)
            tif_file = n.swapaxes(tif_file.reshape(volumes, params["n_ch_tif"], py, px), 0, 1)

            # Filter out planes to analyze
            if params["planes"] is not None:
                tif_file = tif_file[params["planes"]]

            mov_list.append(tif_file)

        # Return mov_list of the current batch
        return mov_list
    
    def _load_faced_tifs(self, paths, params, verbose=True, debug=False, structural=False):
        if structural:
            print("Structural detection not implemented for FACED data!")
            return None
        nz = params["faced_nz"]
        mov_list = []
        for tif_path in paths:
            mov = tifffile.imread(tif_path)
            ny, nx = mov.shape[-2:]
            mov = mov.reshape(-1, nz, ny, nx)
            mov = n.swapaxes(mov, 0, 1).astype(int)
            self.job.log(f"Loaded movie of size: {mov.shape}")
            mov_list.append(mov)
        return mov_list

    def _load_lbm_tifs(self, paths, params, verbose=True, debug=False, structural=False):
        """
        Load tifs that are in the standard lbm imaging format.

        Args:
            paths (list): list of absolute paths to tiff files
            params (dict): parameters for loading the tiffs (inherited from self.job.params in the caller)
            verbose (bool, optional): Verbosity. Defaults to True.
            debug (bool, optional): Debugging mode. Defaults to False.

        Returns:
            _type_: _description_
        """
        todo("Add explanation of how lbm tiffs are organized to the docstring.")
        if structural:
            print("Structural detection not implemented for LBM data!")
            return None

        n_ch_tif = params.get("n_ch_tif", 30)
        if n_ch_tif == 30 and params.get("convert_plane_ids_to_channel_ids", True):
            channels = convert_lbm_plane_to_channel(params["planes"])
        else:
            channels = n.array(params["planes"])
            if params.get("convert_plane_ids_to_channel_ids", True):
                self.job.log(
                    f"Can't convert plane ids to channel ids, because n_ch is set to {n_ch_tif} rather than 30."
                )

        mov_list = []
        for tif_path in paths:
            if verbose:
                self.job.log("Loading %s" % tif_path, 2)

            todo(
                "Removed the **mp_args argument from load_and_stitch_full_tif_mp, should we add it back?"
            )
            im = load_and_stitch_full_tif_mp(
                tif_path,
                channels,
                n_ch_tif,
                filt=params["notch_filt"],
                fix_fastZ=params.get("fix_fastZ", False),
                skip_roi=params.get("skip_roi", None),
                n_proc=params.get("n_proc"),
                verbose=verbose,
                debug=debug,
            )

            mov_list.append(im)

        return mov_list

    def load_roi_start_pix(self, **parameters):
        params = self._update_prms(**parameters)
        if params["lbm"]:
            return self._load_roi_start_pix_lbm(params)
        else:
            return n.array([0]), n.array([0])

    def _load_roi_start_pix_lbm(self, params):
        """
        Get the starting y/x pixels for each ROI in the full image. This is required for stitching ROIs into a full image.

        Args:
            ims: list of numpy arrays, each of shape (nt, ny, nx) where nt is the number of frames, and ny, nx are the pixel dimensions.
            rois: list of dictionaries, each containing the keys 'center' and 'sizeXY' which are the center and size of the ROI in SI units.
        """
        todo(
            "Check if this assumption is correct, that the roi_start_pix values don't depend on which tif we use."
        )
        # we only ever need one of the tifs for this (and it shouldn't matter which one)
        tif_path = self.job.tifs[0]
        roi_start_pix_y, roi_start_pix_x = get_roi_start_pix(tif_path, params)
        return roi_start_pix_y, roi_start_pix_x
