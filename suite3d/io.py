from warnings import warn
import tifffile
import numpy as n
import time
from .utils import todo

# TODO: Implement central filtering method here
# from scipy import signal
# def get_filter(filt_params):
#     """use scipy to get b, a parameters for a notch filter."""
#     return signal.iirnotch(filt_params['f0'],filt_params['Q'], filt_params['line_freq'] )

# def notch_filter(data, filt_params, debug=False):
#     b, a = get_filter(filt_params)
#     line_mov = data[:, ch_id].mean(axis=-1)
#     line_mov_filt = signal.filtfilt(b,a, line_mov)
#     line_mov_diff = line_mov - line_mov_filt

#     tiffile[:, ch_id] = tiffile[:, ch_id]  - line_mov_diff[:,:,n.newaxis]
#     if debug: print('%d filtered in %.2f' % (ch_id, time.time() - tic))
#     tic = time.time()


class IO:
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

    def _get_dataloader(self, params):
        """
        A function that returns the appropriate loader function based on the job's parameters.
        """
        if params["lbm"]:
            return self._load_lbm_tifs
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
        warn("This function is not implemented yet. Please implement it before using it!")
        return mov

    def load_data(self, paths, planes, **parameters):
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
        data = _dataloader(paths, planes, **params)
        return self._preprocess_data(data, params)

    def _load_scanimage_tifs(self, paths, params, filt=None, log_cb=None, verbose=True, debug=False):
        """
        Load tifs that are in the format output by ScanImage.

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
            filt (tuple, optional): parameters of spatiotemporal filter. Defaults to None.
            log_cb (func, optional): Callback function for logging. Defaults to self.job.log.
            verbose (bool, optional): Verbosity. Defaults to True.
            debug (bool, optional): Debugging mode. Defaults to False.
            
        Returns:
            mov (ndarray): the loaded tiff data with shape (planes, frames, y-pixels, x-pixels)
        """
        log_cb = log_cb or self.job.log

        todo("Still need to implement a filtering method!")
        todo("Performance may be improved by using multithreaded operations. Check if this helps.")

        if any([p < 0 or p > params["n_ch_tif"] for p in params["planes"]]):
            raise ValueError(f"Planes must be in range 0-{params['n_ch_tif']}, but it's set to: {params["planes"]}")

        tic = time.time()

        mov_list = []
        for itif, tif_path in enumerate(paths):
            if verbose:
                log_cb(f"Loading tiff {itif+1}/{len(paths)}: {tif_path}", 2)

            tif_file = tifffile.imread(tif_path)
            if params["num_colors"] > 1:
                # TODO: make sure that tiffs are 3d when num_colors==1
                # get functional channel from multi-channel tiff
                if len(tif_file.shape) != 4:
                    raise ValueError(f"tiff file is {tif_file.ndim}D instead of 4D, expecting (frames, colors, y-pixels, x-pixels)")
                if tif_file.shape[1] != params["num_colors"]:
                    raise ValueError(f"tiffs have {tif_file.shape[1]} color channels, expecting {params["num_colors"]}")

                # in general, imaging is only done with one functional color channel, so we take that one and ignore the others
                # if anyone is using multiple functional color channels, they need to modify the code themselves or raise an 
                # issue to ask for this feature to be implemented. A simple work around is to run the suite3d pipeline multiple
                # times with different functional color channels and then combine the results however you see fit. 
                tif_file = n.take(tif_file, params["functional_color_channel"], axis=1)

            assert tif_file.ndim == 3, "tiff file (potentially post color_channel selection) is not 3D, expecting (frames, y-pixels, x-pixels)"

            t, py, px = tif_file.shape
            frames = t // params["n_ch_tif"]
            if frames * params["n_ch_tif"] != t:
                if verbose:
                    extra_planes = t % params["n_ch_tif"]
                    log_cb("Standard 2P Warning: number of planes does not divide into number of tiff images, dropping %d frames" % extra_planes)

                # handle the possibility of uneven plane number by removing extra frames
                tif_file = tif_file[: frames * params["n_ch_tif"]]
                t = frames * len(params["planes"])

            assert frames * len(params["planes"]) == t, "number of planes does not divide into number of tiff images"
            tif_file = n.swapaxes(tif_file.reshape(frames, len(params["planes"]), py, px), 0, 1)
            tif_file = tif_file[params["planes"]]

            if debug:
                print(f"loaded tif_file has shape: {tif_file.shape}, corresponding to [planes, frames, y-pixels, x-pixels]")
                print(f":Loading time up to tiff #{itif+1}: {time.time() - tic:.4f} s")

            mov_list.append(tif_file)

        # concatenate across time to make a single movie
        mov = n.concatenate(mov_list, axis=1)

        if verbose:
            size = mov.nbytes / (1024**3)
            log_cb(f"Loaded {len(paths)} files, total {size:.2f} GB", 1)

        return mov

    def load_and_stitch_tifs(
        self,
        paths,
        planes,
        verbose=True,
        n_proc=15,
        mp_args={},
        filt=None,
        concat=True,
        n_ch=30,
        fix_fastZ=False,
        convert_plane_ids_to_channel_ids=True,
        log_cb=None,
        debug=False,
        lbm=True,
        num_colors=1,
        functional_color_channel=0,
    ):
        """
        Load tifs into memory

        Args:
            paths (list): list of absolute paths to tiff files
            planes (list): planes to load, 0 being deepest and 30 being shallowest
            verbose (bool, optional): Verbosity. Defaults to True.
            n_proc (int, optional): Number of processors. Defaults to 15.
            mp_args (dict, optional): args to pass to worker. Defaults to {}.
            filt (tuple, optional): parameters of spatiotemporal filter. Defaults to None.
            concat (bool, optional): Concatenate across time. Defaults to True.
            convert_plane_ids_to_channel_ids (bool, optional): Convert the plane_ids to account for scanimage ordering. Set to False to access planes by their scanimage channel ID. Defaults to True.
            log_cb (func, optional): Callback function for logging. Defaults to default_log.
            debug (bool, optional): Debugging mode. Defaults to False.
            lbm (bool, optional): whether to load LBM-like tiffs. Defaults to True.
            num_colors (int, optional): number of color channels, only used if loading non LBM data. Defaults to 1.
            functional_color_channel (int, optional): index of functional color channel, only used if loading non LBM data. Defaults to 0.

        Returns:
            _type_: _description_
        """
        if not (lbm):
            # use
            return load_and_stitch_tifs_notLBM(
                paths,
                planes,
                verbose=verbose,
                filt=filt,
                concat=concat,
                log_cb=log_cb,
                debug=debug,
                num_colors=num_colors,
                functional_color_channel=functional_color_channel,
            )

    def load_accessory_data(self, *args, **kwargs):
        """
        Sometimes information related to IO needs to be loaded that is pulled from extra
        outputs of the ``lbmio.load_and_stitch_tifs()`` function. This function (and maybe
        a few others) is/are meant to be called in place of those for efficient processing
        and clear responsibility.

        For example, ``init_pass.run_init_pass()`` calls the following line:
        __, xs = lbmio.load_and_stitch_full_tif_mp(..., get_roi_start_pix=True, ...)

        Which is used to retrieve just "xs"=the starting offsets for each imaging ROI.

        This can be replaced with a new "load accesory data" function that is called
        for this purpose -- depending on preference we can either have it run processing
        again or potentially cache certain results that are known to be needed during an
        initial call to ``load_tifs()``.
        """

    def all_supporting_functions(self, *args, **kwargs):
        """
        A collection of functions that are used to support the main functions of this class.
        These functions are meant to be called by the main functions, and should not be called
        directly by the user.

        Examples:
        load_and_stitch_full_tif_mp
        stitch_rois_fast
        etc...
        """
