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

    def load_data(self, paths, verbose=True, debug=False, **parameters):
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
        mov_list = _dataloader(paths, params, verbose=verbose, debug=debug)
        # print("LOADED")
        # concatenate movies across time to make a single movie
        mov = n.concatenate(mov_list, axis=1)
        # print("CONCATENATED")

        if verbose:
            size = mov.nbytes / (1024**3)
            self.job.log(f"Loaded {len(paths)} files, total {size:.2f} GB", 1)

        return self._preprocess_data(mov, params)

    def _load_scanimage_tifs(self, paths, params, verbose=True, debug=False):
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

        Returns:
            mov (ndarray): the loaded tiff data with shape (planes, frames, y-pixels, x-pixels)
        """
        todo(
            "The lbm loader filters across the slow y-axis (?), should we do something similar here?"
        )
        todo("Performance may be improved by using multithreaded operations.")

        if any([p < 0 or p > params["n_ch_tif"] for p in params["planes"]]):
            raise ValueError(
                f"Planes must be in range 0-{params['n_ch_tif']}, but it's set to: {params['planes']}"
            )

        tic = time.time()

        mov_list = []
        for itif, tif_path in enumerate(paths):
            if verbose:
                self.job.log(f"Loading tiff {itif+1}/{len(paths)}: {tif_path}", 2)

            tif_file = tifffile.imread(tif_path)
            if params["num_colors"] > 1:
                # TODO: make sure that tiffs are 3d when num_colors==1
                # get functional channel from multi-channel tiff
                if len(tif_file.shape) != 4:
                    raise ValueError(
                        f"tiff file is {tif_file.ndim}D instead of 4D, expecting (frames, colors, y-pixels, x-pixels)"
                    )
                if tif_file.shape[1] != params["num_colors"]:
                    raise ValueError(
                        f"tiffs have {tif_file.shape[1]} color channels, expecting {params['num_colors']}"
                    )

                # in general, imaging is only done with one functional color channel, so we take that one and ignore the others
                # if anyone is using multiple functional color channels, they need to modify the code themselves or raise an
                # issue to ask for this feature to be implemented. A simple work around is to run the suite3d pipeline multiple
                # times with different functional color channels and then combine the results however you see fit.
                tif_file = n.take(tif_file, params["functional_color_channel"], axis=1)

            assert (
                tif_file.ndim == 3
            ), "tiff file (potentially post color_channel selection) is not 3D, expecting (frames, y-pixels, x-pixels)"

            t, py, px = tif_file.shape
            frames = t // params["n_ch_tif"]
            if frames * params["n_ch_tif"] != t:
                if verbose:
                    extra_planes = t % params["n_ch_tif"]
                    self.job.log(
                        "Standard 2P Warning: number of planes does not divide into number of tiff images, dropping %d frames"
                        % extra_planes
                    )

                # handle the possibility of uneven plane number by removing extra frames
                tif_file = tif_file[: frames * params["n_ch_tif"]]
                t = frames * len(params["planes"])

            assert (
                frames * len(params["planes"]) == t
            ), "number of planes does not divide into number of tiff images"
            tif_file = n.swapaxes(
                tif_file.reshape(frames, len(params["planes"]), py, px), 0, 1
            )
            tif_file = tif_file[params["planes"]]
            todo("integrate the planes param with this param to make it make sense")
            if params["multiplane_2p_use_planes"] is not None:

                tif_file = tif_file[params["multiplane_2p_use_planes"]]

            if debug:
                print(
                    f"loaded tif_file has shape: {tif_file.shape}, corresponding to [planes, frames, y-pixels, x-pixels]"
                )
                print(f":Loading time up to tiff #{itif+1}: {time.time() - tic:.4f} s")

            mov_list.append(tif_file)

        return mov_list

    def _load_faced_tifs(self, paths, params, verbose=True, debug=False):
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

    @deprecated_inputs("mp_args is never set to anything except an empty dictionary")
    def _load_lbm_tifs(self, paths, params, verbose=True, debug=False):
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
