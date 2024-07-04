import tifffile
import numpy as n
import time
from .utils import todo, convert_lbm_plane_to_channel

from multiprocessing import shared_memory, Pool
from scipy import signal
import imreg_dft as imreg
import json

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
        todo("Performance may be improved by using multithreaded operations.")

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

    def _load_lbm_tifs(
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
        todo("Add explanation of how lbm tiffs are organized to the docstring.")
        

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


        NOTE: xs (the second output of the above mentioned function is the only thing
        ever used by callers of load_and_stitch_full_tif_mp -- except for it's primary
        calls in load_and_stitch_tifs()!)
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

def load_and_stitch_tifs(paths, planes, verbose=True,n_proc=15, mp_args = {}, filt=None, concat=True, n_ch=30, fix_fastZ=False, 
                         convert_plane_ids_to_channel_ids = True, log_cb=default_log, debug=False, lbm=True, num_colors=1, functional_color_channel=0):
    '''
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
    '''        
    if n_ch == 30 and convert_plane_ids_to_channel_ids:
        channels = convert_lbm_plane_to_channel(planes)
    else:
        channels = n.array(planes)
        if convert_plane_ids_to_channel_ids:
            log_cb("Less than 30 channels specified, not converting plane ids to channel ids")
            
    if filt is not None:
        filt = get_filter(filt)

    mov_list = []
    
    for tif_path in paths:
        if verbose: log_cb("Loading %s" % tif_path, 2)
        im, px, py = load_and_stitch_full_tif_mp(tif_path, channels=channels, verbose=False, filt=filt, n_ch = n_ch, n_proc=n_proc,debug=debug,use_roi_idxs=use_roi_idxs,fix_fastZ=fix_fastZ, **mp_args)
        mov_list.append(im)
    if concat:
        mov = n.concatenate(mov_list,axis=1)
        size = mov.nbytes/(1024*1024*1024)
    else: 
        mov = mov_list
        size = n.sum([mx.nbytes/(1024**3) for mx in mov])
    if verbose: log_cb("Loaded %d files, total %.2f GB" % (len(paths),size),1)
    return mov


def load_and_stitch_full_tif_mp(path, channels, n_proc=10, verbose=True,n_ch = 30,
                                translations=None, filt = None, debug=False, get_roi_start_pix=False,
                                use_roi_idxs=None, fix_fastZ=False):
    tic = time.time()
    # TODO imread from tifffile has an overhead of ~20-30 seconds before it actually reads the file?
    tiffile = tifffile.imread(path)
    if debug: print(tiffile.shape)
    if len(tiffile.shape) < 4:
        n_t_ch, n1, n2 = tiffile.shape
        if debug: print(n_t_ch, n_ch, int(n_t_ch/n_ch), n1, n2)
        tiffile = tiffile.reshape(int(n_t_ch/n_ch), n_ch, n1,n2)
    if debug: print(tiffile.shape)
    rois = get_meso_rois(path, fix_fastZ=fix_fastZ)
    # print("XXXXXX %.2f" % (tiffile.nbytes / 1024**3))
    sh_mem = shared_memory.SharedMemory(create=True, size=tiffile.nbytes)
    sh_tif = n.ndarray(tiffile.shape, dtype=tiffile.dtype, buffer=sh_mem.buf)
    sh_tif[:] = tiffile[:]
    if debug: print("4, %.4f" % (time.time()-tic))
    sh_mem_name = sh_mem.name
    sh_mem_params = (sh_tif.shape, sh_tif.dtype)

    n_t, n_ch_tif,__,__ = sh_tif.shape
    n_ch = len(channels)

    # split and stitch two frames to figure out the output size
    ims_sample= split_rois_from_tif(tiffile[:2], rois, ch_id=0)
    if debug: print("5, %.4f" % (time.time()-tic))
    # sample_out = stitch_rois(ims_sample, rois, return_coords=False,mean_img=False)
    if get_roi_start_pix:
        return stitch_rois_fast(ims_sample, rois, mean_img=False, get_roi_start_pix=True, use_roi_idxs=use_roi_idxs)
    sample_out, px, py = stitch_rois_fast(ims_sample, rois,mean_img=False, use_roi_idxs=use_roi_idxs)
    if debug: print("6, %.4f" % (time.time()-tic))
    __, n_y, n_x = sample_out.shape
    del tiffile

    shape_out = (n_ch,n_t,n_y,n_x)
    size_out = (n_t*n_ch*n_y*n_x) * sh_tif[0,0,0,0].nbytes
    sh_mem_out = shared_memory.SharedMemory(create=True, size=size_out)
    sh_out = n.ndarray(shape_out, dtype=sh_tif.dtype, buffer=sh_mem_out.buf)    
    sh_out_name = sh_mem_out.name
    sh_out_params = (sh_out.shape, sh_out.dtype)
    # print(sh_out_params)
    if debug: print("7, %.4f" % (time.time()-tic))

    if translations is None:
        translations = n.zeros((n_ch,2))

    prep_tic = time.time()
    if verbose: print("    Loaded file into shared memory in %.2f sec" % (prep_tic - tic))

    p = Pool(processes = n_proc)
    output = p.starmap(load_and_stitch_full_tif_worker, 
                      [(idx,ch_id, rois, sh_mem_name, sh_mem_params, sh_out_name, sh_out_params, translations[idx], filt, use_roi_idxs)\
                        for idx,ch_id in enumerate(channels)])
    proc_tic = time.time()
    if verbose: print("    Workers completed in %.2f sec" % (proc_tic - prep_tic))
    
    if debug: print("8, %.4f" % (time.time()-tic))

    # print(output)
    if verbose: print("    Total time: %.2f sec" % (time.time()-tic))
    sh_mem.close()
    sh_mem.unlink()
    p.close()
    p.terminate()

    im_full = n.zeros(sh_out.shape, sh_out.dtype)
    im_full[:] = sh_out[:]
    sh_mem_out.close()
    sh_mem_out.unlink()

    return im_full, px, py

def load_and_stitch_full_tif_worker(idx, ch_id, rois, sh_mem_name, sh_arr_params, sh_out_name, sh_out_params, translation=None, 
                                    filt=None,use_roi_idxs=None):
    debug=False
    if debug: print("Loading channel %d" % ch_id)
    tic = time.time()

    sh_mem = shared_memory.SharedMemory(sh_mem_name)
    tiffile = n.ndarray(shape=sh_arr_params[0], dtype=sh_arr_params[1], buffer=sh_mem.buf)

    if filt is not None:
        b,a = filt
        line_mov = tiffile[:,ch_id].mean(axis=-1)
        line_mov_filt = signal.filtfilt(b,a, line_mov)
        line_mov_diff = line_mov - line_mov_filt

        tiffile[:,ch_id] = tiffile[:,ch_id]  - line_mov_diff[:,:,n.newaxis]
        if debug: print('%d filtered in %.2f' % (ch_id, time.time() - tic))
        tic = time.time()

    sh_mem_out = shared_memory.SharedMemory(sh_out_name)
    outputs = n.ndarray(shape=sh_out_params[0], dtype=sh_out_params[1], buffer=sh_mem_out.buf)
    
    prep_time = time.time()
    if debug: print(" %d Loaded in %.2f" % (ch_id, prep_time-tic))
    ims = split_rois_from_tif(tiffile, rois, ch_id = ch_id, return_coords=False)
    split_time = time.time()
    if debug: print(" %d Split in %.2f" % (ch_id, split_time-prep_time))
    outputs[idx], __, __ = stitch_rois_fast(ims, rois, mean_img=False, translation=translation, use_roi_idxs=use_roi_idxs)
    stitch_time = time.time()
    if debug: print(" %d Stitch in %.2f" % (ch_id, stitch_time-split_time))
    if debug: print("Channel %d done in %.2f" % (ch_id, time.time()-tic))

    sh_mem.close() # REMOVE ME IF THERE IS AN ERROR
    sh_mem_out.close() # Remove me if there is an error


    return time.time()-tic


def get_meso_rois(tif_path, max_roi_width_pix=145, fix_fastZ=False, debug=False):
    tf = tifffile.TiffFile(tif_path)
    artists_json = tf.pages[0].tags["Artist"].value

    si_rois = json.loads(artists_json)['RoiGroups']['imagingRoiGroup']['rois']

    all_zs = [roi['zs'] for roi in si_rois]

    if type(fix_fastZ) == int:
        z_imaging = fix_fastZ
    elif fix_fastZ:
        z_imaging = list(set(all_zs[0]).intersection(*map(set,all_zs[1:])))[0]
    else:
        z_imaging = 0
    # if fix_fastZ:
    #     z_imaging = tfu.get_fastZ(tif_path)
    # else:
    #     z_imaging = 0
        
    # print(z_imaging)

    rois = []
    warned = False
    for roi in si_rois:
        if debug: print(roi['zs'])
        if type(roi['scanfields']) != list:
            scanfield = roi['scanfields']
        else: 
            z_match = n.where(n.array(roi['zs'])==z_imaging)[0]
            if len(z_match) == 0: continue
            scanfield = roi['scanfields'][z_match[0]]

    #     print(scanfield)
        roi_dict = {}
        roi_dict['uid'] = scanfield['roiUuid']
        roi_dict['center'] = n.array(scanfield['centerXY'])
        roi_dict['sizeXY'] = n.array(scanfield['sizeXY'])
        roi_dict['pixXY'] = n.array(scanfield['pixelResolutionXY'])
        if roi_dict['pixXY'][0] > max_roi_width_pix and not warned:
            # print("SI ROI pix count in x is %d, which is impossible, setting it to %d" % (roi_dict['pixXY'][0],max_roi_width_pix))
            warned=True
            roi_dict['pixXY'][0] = max_roi_width_pix
    #         print(scanfield)
        rois.append(roi_dict)
    #     print(len(roi['scanfields']))

    roi_pixs = n.array([r['pixXY'] for r in rois])
    return rois


def split_rois_from_tif(im, rois, ch_id = 0, return_coords=False):
    nt, np, ny, nx = im.shape
    n_rois = len(rois)
    ys = n.array([roi['pixXY'][1] for roi in rois])
    n_buff = (ny - ys.sum())/(len(rois)-1)
    if int(n_buff) != n_buff: 
        print("WARNING: Buffer between ROIs is calculated as a non-integer from tiff (%.2f pix)" % n_buff )
    n_buff = int(n_buff)
    
    split_ims = []
    coords = []
    y_start = 0
    for i in range(n_rois):
        ny = ys[i]
        split_im = im[:, ch_id, y_start:y_start+ny]
        coord = n.meshgrid(n.arange(y_start, y_start+ny), n.arange(0,im.shape[-1]))
        split_ims.append(split_im)
        y_start += ny + n_buff
        coords.append(coord)
    if return_coords: return split_ims, coords
    return split_ims


def stitch_rois_fast(ims, rois, mean_img=False, translation = None, get_roi_start_pix=False, use_roi_idxs=None):
    tic = time.time()


    if use_roi_idxs is not None:
        ims = [ims[i] for i in use_roi_idxs]
        rois = [rois[i] for i in use_roi_idxs]
    sizes_pix = n.array([im.shape[1:][::-1] for im in ims])

    centers = n.array([r['center'] for r in rois])
    sizes = n.array([r['sizeXY'] for r in rois])
    corners = centers - sizes/2
    n_rois = len(rois)

    # X is the fast axis along the resonant scanner line direction, Y is orthogonal slow axis
    # For a typical strip, x extent is small and y extent is large

    # maximim and minimum x/y coordinates in SI units (not pixels, also strangely not always um)
    xmin, xmax = corners[:,0].min(),(corners[:,0] + sizes[:,0]).max()
    ymin, ymax = corners[:,1].min(),(corners[:,1] + sizes[:,1]).max()

    # calculate pixel sizes (relative to weird SI units)
    pixel_sizes = sizes/sizes_pix
    psize_y = n.mean(pixel_sizes[:,1])
    psize_x = n.mean(pixel_sizes[:,0])
    assert n.product(n.isclose(pixel_sizes[:,1]-psize_y, 0)), "Y pixels not uniform"
    assert n.product(n.isclose(pixel_sizes[:,0]-psize_x, 0)), "X pixels not uniform"

    # SI unit coordinates of each pixel of the full image
    full_xs = n.arange(xmin,xmax, psize_x)
    full_ys = n.arange(ymin,ymax, psize_y)
    # accumulate how many times a value is written into each pixel
#     if accumulate: full_image_acc = n.zeros(full_image.shape[1:])

    # calculate the starting pixel for each of the ROIs when placing them into full image
    roi_start_pix_x = []
    roi_start_pix_y = []
    for roi_idx in range(n_rois):
        x_corner = corners[roi_idx,0]
        y_corner = corners[roi_idx, 1]

        closest_x_idx = n.argmin(n.abs(full_xs-x_corner))
        roi_start_pix_x.append(closest_x_idx)
        closest_x = full_xs[closest_x_idx]
        if not n.isclose(closest_x, x_corner):
            print("ROI %d x does not fit perfectly into image, corner is %.4f but closest available is %.4f" %\
                  (roi_idx, closest_x, x_corner))

        closest_y_idx = n.argmin(n.abs(full_ys-y_corner))
        roi_start_pix_y.append(closest_y_idx)
        closest_y = full_ys[closest_y_idx]
        if not n.isclose(closest_y, y_corner):
            print("ROI %d y does not fit perfectly into image, corner is %.4f but closest available is %.4f" %\
                  (roi_idx, closest_y, y_corner))
            
    prep_tic = time.time()

    if get_roi_start_pix:
        return n.sort(roi_start_pix_y), n.sort(roi_start_pix_x)

    if mean_img:
        n_t = 1
    else:
        n_t = ims[0].shape[0]

    full_image = n.zeros((n_t, len(full_ys), len(full_xs)))

    stitch_rois_fast_helper(full_image, ims, n.array(roi_start_pix_x), n.array(roi_start_pix_y), sizes_pix, mean_img)

    if translation is not None and n.linalg.norm(translation) > 0.01:
        for i in range(full_image.shape[0]):
            full_image[i] = imreg.transform_img(full_image[i],tvec = translation)

    # print(time.time()-prep_tic)
    return full_image, psize_x, psize_y
    
    
# @numba.jit(nopython=True)
def stitch_rois_fast_helper(full_image, ims, roi_start_pix_x, roi_start_pix_y, sizes_pix, mean_img):
    n_rois = len(ims)
    # place each ROI into full image
    for roi_idx in range(n_rois):
        roi_x_start,roi_y_start = roi_start_pix_x[roi_idx], roi_start_pix_y[roi_idx]
        roi_x_end = roi_x_start + sizes_pix[roi_idx][0]
        roi_y_end = roi_y_start + sizes_pix[roi_idx][1]

        if mean_img:
            assert False, 'not implemented'
            # return full_image[0, roi_y_start:roi_y_end, roi_x_start:roi_x_end] = ims[roi_idx].mean(axis=0)
        else:
            full_image[:,roi_y_start:roi_y_end, roi_x_start:roi_x_end] = ims[roi_idx]