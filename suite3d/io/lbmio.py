import tifffile
import numpy as n
import time
import copy
from multiprocessing import shared_memory, Pool
from scipy import signal

# import imreg_dft as imreg
import json
from ..developer import deprecated_inputs, todo


@deprecated_inputs(
    "Translation is never set to anything except None or zeros, so it's effectively ignored."
)
def load_and_stitch_full_tif_mp(
    path,
    channels,
    n_ch_tif,
    filt=None,
    fix_fastZ=False,
    n_proc=10,
    verbose=True,
    debug=False,
    translations=None,  # deprecated
    skip_roi=None,
):
    tic = time.time()

    todo(
        "imread from tifffile has an overhead of ~20-30 seconds before it actually reads the file? Can we speed this up?"
    )
    tiffile = tifffile.imread(path)

    if debug:
        print("from load_and_stitch_full_tif_mp, tifffile has shape: ", tiffile.shape)

    if len(tiffile.shape) < 4:
        n_t_ch, n1, n2 = tiffile.shape
        if debug:
            print(n_t_ch, n_ch_tif, int(n_t_ch / n_ch_tif), n1, n2)
        tiffile = tiffile.reshape(int(n_t_ch / n_ch_tif), n_ch_tif, n1, n2)

    if debug:
        print(
            "from load_and_stitch_full_tif_mp, after wrangling tifffile has shape: ",
            tiffile.shape,
        )

    # print("SKipping " + str(skip_roi))
    rois = get_meso_rois(path, fix_fastZ=fix_fastZ, skip_roi=skip_roi)

    sh_mem = shared_memory.SharedMemory(create=True, size=tiffile.nbytes)
    sh_tif = n.ndarray(tiffile.shape, dtype=tiffile.dtype, buffer=sh_mem.buf)
    sh_tif[:] = tiffile[:]
    if debug:
        print("4, %.4f" % (time.time() - tic))

    sh_mem_name = sh_mem.name
    sh_mem_params = (sh_tif.shape, sh_tif.dtype)
    n_t = sh_tif.shape[0]
    n_ch_tif = len(channels)

    # split and stitch two frames to figure out the output size
    ims_sample = _split_rois_from_tif(tiffile[:2], rois, ch_id=0)
    if debug:
        print("5, %.4f" % (time.time() - tic))
    rois_stitch = copy.deepcopy(rois)
    if skip_roi is not None:
        ims_sample.pop(skip_roi)
        rois_stitch.pop(skip_roi)
    sample_out = _stitch_rois_fast(ims_sample, rois_stitch)
    __, n_y, n_x = sample_out.shape

    if debug:
        print("6, %.4f" % (time.time() - tic))

    del tiffile

    shape_out = (n_ch_tif, n_t, n_y, n_x)
    size_out = (n_t * n_ch_tif * n_y * n_x) * sh_tif[0, 0, 0, 0].nbytes
    sh_mem_out = shared_memory.SharedMemory(create=True, size=size_out)
    sh_out = n.ndarray(shape_out, dtype=sh_tif.dtype, buffer=sh_mem_out.buf)
    sh_out_name = sh_mem_out.name
    sh_out_params = (sh_out.shape, sh_out.dtype)

    if debug:
        print("7, %.4f" % (time.time() - tic))

    if translations is None:
        translations = n.zeros((n_ch_tif, 2))

    if verbose:
        prep_tic = time.time()
        print("    Loaded file into shared memory in %.2f sec" % (prep_tic - tic))

    p = Pool(processes=n_proc)
    _ = p.starmap(
        load_and_stitch_full_tif_worker,
        [
            (
                idx,
                ch_id,
                rois,
                sh_mem_name,
                sh_mem_params,
                sh_out_name,
                sh_out_params,
                translations[idx],
                filt,
                skip_roi,
            )
            for idx, ch_id in enumerate(channels)
        ],
    )

    if verbose:
        proc_tic = time.time()
        print("    Workers completed in %.2f sec" % (proc_tic - prep_tic))

    if debug:
        print("8, %.4f" % (time.time() - tic))

    if verbose:
        print("    Total time: %.2f sec" % (time.time() - tic))

    sh_mem.close()
    sh_mem.unlink()
    p.close()
    p.terminate()

    im_full = n.zeros(sh_out.shape, sh_out.dtype)
    im_full[:] = sh_out[:]
    sh_mem_out.close()
    sh_mem_out.unlink()

    return im_full


@deprecated_inputs(
    "Translation is never set to anything except None or zeros, so it's effectively ignored."
)
def load_and_stitch_full_tif_worker(
    idx,
    ch_id,
    rois,
    sh_mem_name,
    sh_arr_params,
    sh_out_name,
    sh_out_params,
    translation=None,
    filt=None,
    skip_roi=None,
):
    debug = False
    if debug:
        print("Loading channel %d" % ch_id)
    tic = time.time()

    sh_mem = shared_memory.SharedMemory(sh_mem_name)
    tiffile = n.ndarray(shape=sh_arr_params[0], dtype=sh_arr_params[1], buffer=sh_mem.buf)

    if filt is not None:
        b, a = _get_filter(filt)
        line_mov = tiffile[:, ch_id].mean(axis=-1)
        line_mov_filt = signal.filtfilt(b, a, line_mov)
        line_mov_diff = line_mov - line_mov_filt

        tiffile[:, ch_id] = tiffile[:, ch_id] - line_mov_diff[:, :, n.newaxis]
        if debug:
            print("%d filtered in %.2f" % (ch_id, time.time() - tic))
        tic = time.time()

    sh_mem_out = shared_memory.SharedMemory(sh_out_name)
    outputs = n.ndarray(
        shape=sh_out_params[0], dtype=sh_out_params[1], buffer=sh_mem_out.buf
    )

    prep_time = time.time()
    if debug:
        print(" %d Loaded in %.2f" % (ch_id, prep_time - tic))
    ims = _split_rois_from_tif(tiffile, rois, ch_id=ch_id)
    split_time = time.time()
    if debug:
        print(" %d Split in %.2f" % (ch_id, split_time - prep_time))
    rois_stitch = copy.deepcopy(rois)
    if skip_roi is not None:
        ims.pop(skip_roi)
        rois_stitch.pop(skip_roi)
    outputs[idx] = _stitch_rois_fast(ims, rois_stitch)
    stitch_time = time.time()
    if debug:
        print(" %d Stitch in %.2f" % (ch_id, stitch_time - split_time))
    if debug:
        print("Channel %d done in %.2f" % (ch_id, time.time() - tic))

    sh_mem.close()  # REMOVE ME IF THERE IS AN ERROR
    sh_mem_out.close()  # Remove me if there is an error

    return time.time() - tic


def get_meso_rois(
    tif_path, max_roi_width_pix=145, fix_fastZ=False, debug=False, skip_roi=None
):
    tf = tifffile.TiffFile(tif_path)
    artists_json = tf.pages[0].tags["Artist"].value

    si_rois = json.loads(artists_json)["RoiGroups"]["imagingRoiGroup"]["rois"]
    all_zs = [roi["zs"] for roi in si_rois]
    if debug:
        print(all_zs)

    if type(fix_fastZ) == int:
        z_imaging = fix_fastZ
    elif fix_fastZ:
        z_imaging = list(set(all_zs[0]).intersection(*map(set, all_zs[1:])))[0]
    else:
        z_imaging = 0

    rois = []
    warned = False
    # print([s["zs"] for s in si_rois])
    if skip_roi is not None:
        si_rois.pop(skip_roi)
        if type(z_imaging) == list:
            z_imaging.pop(skip_roi)
    # print([s["zs"] for s in si_rois])
    for roi in si_rois:
        if debug:
            print(roi["zs"])
        if type(roi["scanfields"]) != list:
            scanfield = roi["scanfields"]
        else:
            z_match = n.where(n.array(roi["zs"]) == z_imaging)[0]
            if len(z_match) == 0:
                continue
            scanfield = roi["scanfields"][z_match[0]]

        roi_dict = {}
        # print(scanfield)
        roi_dict["uid"] = scanfield["roiUuid"]
        roi_dict["center"] = n.array(scanfield["centerXY"])
        roi_dict["sizeXY"] = n.array(scanfield["sizeXY"])
        roi_dict["pixXY"] = n.array(scanfield["pixelResolutionXY"])
        if roi_dict["pixXY"][0] > max_roi_width_pix and not warned:
            # print("SI ROI pix count in x is %d, which is impossible, setting it to %d" % (roi_dict['pixXY'][0],max_roi_width_pix))
            warned = True
            roi_dict["pixXY"][0] = max_roi_width_pix
        rois.append(roi_dict)
    # if skip_roi is not None:
    # rois.pop(skip_roi)
    return rois


def _split_rois_from_tif(im, rois, ch_id=0):
    """
    Split the tiff image into ROIs based on the rois dictionary.

    The image will have shape (nt, np, ny, nx) where nt is the number of frames, np is the number of planes, and ny, nx are the pixel dimensions.
    ROIs are split along the Y axis, with number of pixels of the i'th ROI defined by rois[i]['pixXY'][1].
    There is a buffer of n_buff pixels between each ROI along the y-axis -- this is calculated from the tiff image size and should be an integer.

    The function simply splits the image along the Y axis, ignoring the buffer, and returns a list of images, one for each ROI.
    """
    ny = im.shape[2]
    n_rois = len(rois)
    ys = n.array([roi["pixXY"][1] for roi in rois])
    n_buff = (ny - ys.sum()) / (len(rois) - 1)
    if int(n_buff) != n_buff:
        print(
            "WARNING: Buffer between ROIs is calculated as a non-integer from tiff (%.2f pix)"
            % n_buff
        )
    n_buff = int(n_buff)

    split_ims = []
    y_start = 0
    for i in range(n_rois):
        ny = ys[i]
        split_im = im[:, ch_id, y_start : y_start + ny]
        split_ims.append(split_im)
        y_start += ny + n_buff

    return split_ims


def get_roi_start_pix(tif_path, params):
    """
    Get the starting y/x pixels for each ROI in the full image. This is required for stitching ROIs into a full image.

    This function is called from outside the standard load_data pipeline for registration and other uses. It depends on
    _get_roi_start_pix, which actually does the work of calculating the starting pixel positions for each ROI.

    Args:
        tif_path: path to the tiff file to compute starting ROIs on
        params: dictionary of parameters, including fix_fastZ, which is a boolean indicating whether to fix the fastZ plane to the first plane.
                should be from job.params!
    """
    tiffile = tifffile.imread(tif_path)
    if len(tiffile.shape) < 4:
        n_t_ch, n1, n2 = tiffile.shape
        n_ch = params["n_ch_tif"]
        tiffile = tiffile.reshape(int(n_t_ch / n_ch), n_ch, n1, n2)
    rois = get_meso_rois(
        tif_path,
        fix_fastZ=params.get("fix_fastZ", False),
        skip_roi=params.get("skip_roi", None),
    )
    # print(len(tiffile))
    ims_sample = _split_rois_from_tif(tiffile[:2], rois, ch_id=0)
    if params["skip_roi"] is not None:
        rois.pop(params["skip_roi"])
        print("Popping ROI %d" % params["skip_roi"])
        ims_sample.pop(params["skip_roi"])
    roi_start_pix_y, roi_start_pix_x = _get_roi_start_pix(
        ims_sample,
        rois,
        return_full=False,
    )



    return roi_start_pix_y, roi_start_pix_x


def _get_roi_start_pix(ims, rois, return_full=False):
    """
    Get the starting y/x pixels for each ROI in the full image. This is required for stitching ROIs into a full image.

    Args:
        ims: list of numpy arrays, each of shape (nt, ny, nx) where nt is the number of frames, and ny, nx are the pixel dimensions.
        rois: list of dictionaries, each containing the keys 'center' and 'sizeXY' which are the center and size of the ROI in SI units.
    """
    # print(len(ims))
    # print(ims[0].shape)
    xpix_size = [im.shape[2] for im in ims]
    ypix_size = [im.shape[1] for im in ims]
    sizes_pix = n.stack((xpix_size, ypix_size), axis=1)

    centers = n.array([r["center"] for r in rois])
    sizes = n.array([r["sizeXY"] for r in rois])
    corners = centers - sizes / 2
    # print(corners)

    # X is the fast axis along the resonant scanner line direction, Y is orthogonal slow axis
    # For a typical strip, x extent is small and y extent is large

    # maximim and minimum x/y coordinates in SI units (not pixels, also strangely not always um)
    xmin = corners[:, 0].min()
    xmax = (corners[:, 0] + sizes[:, 0]).max()
    ymin = corners[:, 1].min()
    ymax = (corners[:, 1] + sizes[:, 1]).max()

    # calculate pixel sizes (relative to weird SI units)
    pixel_sizes = sizes / sizes_pix
    psize_y = n.mean(pixel_sizes[:, 1])
    psize_x = n.mean(pixel_sizes[:, 0])
    assert n.prod(n.isclose(pixel_sizes[:, 1] - psize_y, 0)), "Y pixels not uniform"
    assert n.prod(n.isclose(pixel_sizes[:, 0] - psize_x, 0)), "X pixels not uniform"

    # SI unit coordinates of each pixel of the full image
    full_xs = n.arange(xmin, xmax, psize_x)
    full_ys = n.arange(ymin, ymax, psize_y)

    def _get_start_pix_from_corner(corners, full_coords, x_or_y=None):
        closest_idx = n.argmin(
            n.abs(full_coords.reshape(1, -1) - corners.reshape(-1, 1)), axis=1
        )
        closest_coord = full_coords[closest_idx]
        if not n.isclose(closest_coord, corners).all():
            _insert = ""
            if x_or_y == "x" or x_or_y == "y":
                _insert = f" on the {x_or_y} axis"
            print(
                f"ROI does not fit perfectly into image{_insert}, corner is {corners} but closest available is {closest_coord}"
            )
        return closest_idx

    # roi_start_pix_x = n.sort(_get_start_pix_from_corner(corners[:, 0], full_xs, "x"))
    # roi_start_pix_y = n.sort(_get_start_pix_from_corner(corners[:, 1], full_ys, "y"))

    # 2024-08-06 SS003 needed this to work
    roi_start_pix_x = _get_start_pix_from_corner(corners[:, 0], full_xs, "x")
    roi_start_pix_y = _get_start_pix_from_corner(corners[:, 1], full_ys, "y")

    # I removed the sorting from here, not explicitly printing a warning because it spams
    # im not sure if this breaks things or not... AH
    if ~n.all(n.diff(roi_start_pix_x) > 0) or (~n.all(roi_start_pix_y) > 0):
        warn = True

    if return_full:
        return dict(
            roi_start_pix_x=roi_start_pix_x,
            roi_start_pix_y=roi_start_pix_y,
            full_xs=full_xs,
            full_ys=full_ys,
            psize_x=psize_x,
            psize_y=psize_y,
            sizes_pix=sizes_pix,
        )

    return roi_start_pix_y, roi_start_pix_x


@deprecated_inputs(
    "Translation is never set to anything except None or zeros, so it's effectively ignored."
)
def _stitch_rois_fast(
    ims,
    rois,
):
    roi_positions = _get_roi_start_pix(ims, rois, return_full=True)

    roi_start_pix_x = roi_positions["roi_start_pix_x"]
    roi_start_pix_y = roi_positions["roi_start_pix_y"]
    full_xs = roi_positions["full_xs"]
    full_ys = roi_positions["full_ys"]
    sizes_pix = roi_positions["sizes_pix"]

    # place each ROI into full image
    n_rois = len(ims)
    full_image = n.zeros((ims[0].shape[0], len(full_ys), len(full_xs)))
    for roi_idx in range(n_rois):
        roi_x_start, roi_y_start = roi_start_pix_x[roi_idx], roi_start_pix_y[roi_idx]
        roi_x_end = roi_x_start + sizes_pix[roi_idx][0]
        roi_y_end = roi_y_start + sizes_pix[roi_idx][1]

        full_image[:, roi_y_start:roi_y_end, roi_x_start:roi_x_end] = ims[roi_idx]
    return full_image


def _get_lbm_plane_to_channel():
    """lbm planes are ordered in a non-contiguous fashion. This function returns the channel numbers for the planes."""
    return (
        n.array(
            [
                1,
                5,
                6,
                7,
                8,
                9,
                2,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                3,
                18,
                19,
                20,
                21,
                22,
                23,
                4,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
            ]
        )
        - 1
    )


def convert_lbm_plane_to_channel(planes):
    """lbm planes are ordered in a non-contiguous fashion. This function converts the plane numbers to channel numbers."""
    return _get_lbm_plane_to_channel[n.array(planes)]


def convert_lbm_channel_to_plane(channels):
    """lbm planes are ordered in a non-contiguous fashion. This function converts the channel numbers to plane numbers."""
    lbm_ch_to_plane = n.array(n.argsort(_get_lbm_plane_to_channel()))
    return lbm_ch_to_plane[n.array(channels)]


def _get_filter(filt_params):
    """use scipy to get b, a parameters for a notch filter."""
    if filt_params is not None:
        return signal.iirnotch(
            filt_params["f0"], filt_params["Q"], filt_params["line_freq"]
        )
    return None
