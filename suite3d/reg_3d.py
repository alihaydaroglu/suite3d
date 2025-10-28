# new function used for the 3d registration
import os
import numpy as n

from .reference_image import HAS_CUPY

np = n

try:
    import cupy as cp
    from cupyx.scipy import fft as cufft
    from cupyx.scipy import ndimage as cuimage
    HAS_CUPY = True
except ImportError:
    import numpy as cp
    from scipy.fft import fft as cufft
    from scipy import ndimage as cuimage
    HAS_CUPY = False

import scipy
from numba import njit

from . import reference_image as ref
from . import register_gpu as reg
from . import utils


@njit(nogil=True, cache=True)
def shift_mov_lbm_fast(mov, plane_shifts, fill_value=0):
    """
    Apply LBM shifts over a 4D movie, these shifts are same for all time and have different x/y shifts per z-plane

    Parameters
    ----------
    mov : nd array (nz, nt, ny, nx)
        Movie to be shifted
    plane_shifts : ndarray ( nz, 2)
        The (y_shift, x_shift) per z-plane to allign the planes
    fill_value : optional
        The vaule to fill the empty spaces caused by shifting the array, by default 0

    Returns
    -------
    ndarray (nz, nt, ny, nx)
        The shifted array
    """

    shifted_mov = np.zeros_like(mov)
    nz, __, __, __ = mov.shape

    for z in range(nz):
        shift = plane_shifts[z, :]
        # print(z, shift)
        if (shift[0] == 0) & (shift[1] == 0):
            # print(1)
            shifted_mov[z, :, :, :] = mov[z, :, :, :]
        elif (shift[0] >= 0) & (shift[1] >= 0):
            # print(2)
            if shift[1] > 0:
                # print('a')
                shifted_mov[z, :, :, : shift[1]] = fill_value
            if shift[0] > 0:
                # print('b')
                shifted_mov[z, :, : shift[0], :] = fill_value

            if shift[0] == 0:
                # print('c')
                shifted_mov[z, :, :, shift[1] :] = mov[z, :, :, : -shift[1]]
            elif shift[1] == 0:
                shifted_mov[z, :, shift[0] :, :] = mov[z, :, : -shift[0], :]
            else:
                # print('e')
                shifted_mov[z, :, shift[0] :, shift[1] :] = mov[
                    z, :, : -shift[0], : -shift[1]
                ]
        elif (shift[0] >= 0) & (shift[1] < 0):
            # print(3)
            shifted_mov[z, :, :, shift[1] :] = fill_value
            if shift[0] > 0:
                shifted_mov[z, :, : shift[0], :] = fill_value
                shifted_mov[z, :, shift[0] :, : shift[1]] = mov[
                    z, :, : -shift[0], -shift[1] :
                ]
            else:
                shifted_mov[z, :, shift[0] :, : shift[1]] = mov[z, :, :, -shift[1] :]

        elif (shift[0] < 0) & (shift[1] >= 0):
            # print(4)
            shifted_mov[z, :, shift[0] :, :] = fill_value
            if shift[1] > 0:
                shifted_mov[z, :, :, : shift[1]] = fill_value
                shifted_mov[z, :, : shift[0], shift[1] :] = mov[
                    z, :, -shift[0] :, : -shift[1]
                ]
            else:
                shifted_mov[z, :, : shift[0], shift[1] :] = mov[z, :, -shift[0] :, :]

        else:
            # print(5)
            shifted_mov[z, :, :, shift[1] :] = fill_value
            shifted_mov[z, :, shift[0] :, :] = fill_value
            shifted_mov[z, :, : shift[0], : shift[1]] = mov[
                z, :, -shift[0] :, -shift[1] :
            ]

    return shifted_mov


@njit(parallel=True)
def mult_and_normailise(fft1, fft2):
    """
    Multiplies and normalises to arrays, Not currently used. Currently steps applied seperatley as normalised refernce fft is
    pre-calculated.

    Parameters
    ----------
    fft1 : ndarray
        array 1
    fft2 : ndarray
        array 2

    Returns
    -------
    ndarray
        multiplied and normalised array
    """
    return (fft1 * fft2) / np.abs(fft1 * fft2)


@njit(parallel=True, cache=True)
def mult_fft(fft1, fft2):
    """
    Multiplies the fft'd arrays, in a fast numba fashion

    Parameters
    ----------
    fft1 : ndarray
        array 1
    fft2 : ndarray
        array 2

    Returns
    -------
    ndarray
        multiplied array
    """
    return fft1 * fft2


@njit(parallel=True, cache=True)
def div_norm_fft(fft):
    """
    Normalise a array.

    Parameters
    ----------
    fft : ndarray
        array to be normalised

    Returns
    -------
    ndarray
        normalised array
    """
    fft = fft / (1e-5 + np.absolute(fft))
    return fft


def est_sub_pixel_shift(r, np=np):
    """
    Estimates the subpixel shift of the phase correlation.
    The estimate uses the difference between the phasecoreraltion adjacent to the peak, normalised by the total their
    difference from the peak.

    Parameters
    ----------
    r : ndarray (nS)
        nS - any spatial dimension, a 1D line of the phase correlation going through the peak of the 3D phase correaltion
    np : package, optional
        can input cp to get this function on gpu, by default np

    Returns
    -------
    float
        The estimates sub pixel shift for the inputed dimension
    """
    center = np.argmax(r)
    max = r.shape[0]

    # returns the shift est so result is -max/2 to max/2
    shift = center - np.floor(max / 2)  # argmax starts from 0

    # (center + 1) % max is so if the shift is -1 therfore argmax is last idx, need center+1 to loop back around.
    sub_pixel = (r[(center + 1) % max] - r[center - 1]) / (
        2 * r[center] - r[center - 1] - r[(center + 1) % max]
    )
    return shift + sub_pixel * 0.5


def process_phase_corr_per_frame(phase_corr, pc_size):
    """
    Analysise the phase correlation to return useful information, a re-aranged phase_corr, peak location and
    integer + sub pixel shifts.
    This function is used for the cpu where frames are done sequentially

    Parameters
    ----------
    phase_corr : ndarray (nz, ny, nx)
        The full phase correlation for a frame
    pc_size : ndarray (nz_pc, ny_pc, nx_pc)
        This determines the size of the re-aranged phase correlation array and the maximum size of shifts allowed

    Returns
    -------
    phase_corr_shifted : ndarray (2*nz_pc +1, 2*ny_pc + 1, 2*nx_pc + 1)
        The phase correlation cropped and shifted so the peak is central
    shift : ndarray (3,)
        The integer shift to maximise phase correlation
    pc_peak_lock : ndarray (3,)
        The index of the maximum value of the shift phase correlation array
    sub_pixel_shifts : ndarray (3,)
        The sub pixel shift estiamted from the phase correlation

    """

    max_pc_size = pc_size * 2 + 1
    nz, ny, nx = phase_corr.shape
    phase_corr_shifted = np.zeros((max_pc_size[0], max_pc_size[1], max_pc_size[2]))

    # for example:
    # want z planes 0,1,2 to go to 2,3,4
    # want z planes 14,13 to go to 1,0
    # so the new z plane -2 is the shift!
    # as for x/y 0-50 goes to 50-101
    # and the last 50 go to 0-50

    # have z+/- x+/- y+/-
    # add z+ x+ y+
    phase_corr_shifted[pc_size[0] :, pc_size[1] :, pc_size[2] :] = phase_corr[
        : pc_size[0] + 1, : pc_size[1] + 1, : pc_size[2] + 1
    ]
    # add z+ x- y+
    phase_corr_shifted[pc_size[0] :, pc_size[1] :, : pc_size[2]] = phase_corr[
        : pc_size[0] + 1, : pc_size[1] + 1, nx - pc_size[2] :
    ]
    # add z+ x+ y-
    phase_corr_shifted[pc_size[0] :, : pc_size[1], pc_size[2] :] = phase_corr[
        : pc_size[0] + 1, ny - pc_size[1] :, : pc_size[2] + 1
    ]
    # add z+ x- y-
    phase_corr_shifted[pc_size[0] :, : pc_size[1], : pc_size[2]] = phase_corr[
        : pc_size[0] + 1, ny - pc_size[1] :, nx - pc_size[2] :
    ]

    # add z- x+ y+
    phase_corr_shifted[: pc_size[0], pc_size[1] :, pc_size[2] :] = phase_corr[
        nz - pc_size[0] :, : pc_size[1] + 1, : pc_size[2] + 1
    ]
    # add z- x- y+
    phase_corr_shifted[: pc_size[0], pc_size[1] :, : pc_size[2]] = phase_corr[
        nz - pc_size[0] :, : pc_size[1] + 1, nx - pc_size[2] :
    ]
    # add z- x+ y-
    phase_corr_shifted[: pc_size[0], : pc_size[1], pc_size[2] :] = phase_corr[
        nz - pc_size[0] :, ny - pc_size[1] :, : pc_size[2] + 1
    ]
    # add z- x- y-
    phase_corr_shifted[: pc_size[0], : pc_size[1], : pc_size[2]] = phase_corr[
        nz - pc_size[0] :, ny - pc_size[1] :, nx - pc_size[2] :
    ]

    shift = np.zeros(3)
    pc_peak_loc = np.zeros(3, dtype=np.int16)

    mx = np.argmax(phase_corr_shifted)
    pc_peak_loc[:] = np.unravel_index(mx, phase_corr_shifted.shape)
    shift[:] = pc_peak_loc[:] - pc_size

    z_sub_pixel = est_sub_pixel_shift(
        phase_corr_shifted[:, pc_peak_loc[1], pc_peak_loc[2]]
    )
    x_sub_pixel = est_sub_pixel_shift(
        phase_corr_shifted[pc_peak_loc[0], :, pc_peak_loc[2]]
    )
    y_sub_pixel = est_sub_pixel_shift(
        phase_corr_shifted[pc_peak_loc[0], pc_peak_loc[1], :]
    )

    sub_pixel_shifts = [z_sub_pixel, y_sub_pixel, x_sub_pixel]
    return phase_corr_shifted, shift, pc_peak_loc, sub_pixel_shifts


# TODOmove/integrate into reference.py
def gaussian_fft3D(sig, nZ, nY, nX):
    """
    Returns a gaussian filter in the Fourier domain std sig and size (nY, nX).
    This function is adapted from suite 2p

    NOTE - this function is currently set up NOT apply smoothing over z-axis

    Parameters
    ----------
    sig: float
        standard deviation of the gaussian
    nY: int
        length of the y axis
    nX: int
        length of the x axis

    Returns
    -------
    fhg: ndarray
        gussian filter in the Fourier domain
    """

    # need 2D x/y mesh grid
    zz, yy, xx = ref.mean_centered_meshgrid3D(nZ, nY, nX)

    hgx = n.exp(-n.square(xx / sig) / 2)
    hgy = n.exp(-n.square(yy / sig) / 2)
    # Not smoothing over z
    # hgz = n.exp(-n.square(zz/(0.5 * sig)) / 2)

    hgg = hgy * hgx  # * hgz
    hgg /= hgg.sum()
    fhg = n.real(scipy.fft.fftn(n.fft.ifftshift(hgg)))

    # make it uniform over z-axis
    fhg[1:, :, :] = fhg[0, :, :]
    return fhg


@njit(parallel=True, cache=True)
def apply_mask4D(mov, mask_mul, mask_offset, out):
    """
    Appleis the multiplcation and addition mask in numba parallel manner
    Parameters
    ----------
    mov : ndarray (nz, nt, ny, nx)
        The movie to be masked
    mask_mul : ndarray (nz, ny, nx)
        The multiplcation mask
    mask_offset : ndarray (nz, ny, nx)
        The addition/offset mask
    out : ndarray (nz, nt, ny, nx)
        A empty array the same size as the movie which the masked movie is saved to

    Returns
    -------
    ndarray (nz, nt, ny, nx)
        The movie with the multiplcation and addition masks applied
    """
    for t in range(mov.shape[1]):
        out[:, t, :, :] = mov[:, t, :, :] * mask_mul + mask_offset
    return out


def apply_mask3D(data, mask_mul, mask_offset):
    # print("APPYING MASK")
    # print(data.shape)
    # print(mask_mul.shape)
    # print(mask_offset.shape)
    return data * mask_mul + mask_offset


def mask_filter_fft_ref(ref_img, mult_mask, add_mask, smooth=0.5):
    """
    Mask, filter and fourier transform the 3D reference image, should be done as part of the reference img creation

    Parameters
    ----------
    ref_img : ndarray (nz, ny, nx)
        The calculated reference image
    mult_mask : ndarray (nz, ny, nx)
        The multiplication mask/ hammingwindow used for registration
    add_mask : ndarray (nz, ny, nx)
        The addition, offset mask used for registration
    smooth : float
        The value of sigma used for the gaussian filter, this is then fft'd
        0 means no smothing, <0.5 little smoothing, <1 moderate smoothing, 1+ large smoothing ,by default 0.5

    Returns
    -------
    ndarray (nz, ny, nx)
        The masked,filtered fourier transformed reference image
    """

    nz, ny, nx = ref_img.shape

    masked_ref = apply_mask3D(ref_img, mult_mask, add_mask)

    # take 3D fourier transform
    fft_3d_ref = scipy.fft.fftn(masked_ref)
    fft_3d_ref_conj = np.conj(fft_3d_ref)
    fft_3d_ref_conj = div_norm_fft(fft_3d_ref_conj)

    # apply the gaussian filter fft done like sute 2p registration
    # currently gaussianfft shift is not set up to smooth in the z axis
    gaussian_fillter_fftd = gaussian_fft3D(smooth, nz, ny, nx)
    fft_3d_ref_conj *= gaussian_fillter_fftd

    return fft_3d_ref_conj


def clip_mov_cpu(mov, rmin, rmax):
    """
    Clip the movie per plane between rmin and rmax, numba seems to be slower

    Parameters
    ----------
    mov : ndarray (nz, nt, ny, nx)
        The movie to be clipped
    rmin : ndarray (nz,)
        The minimum allowed value for each plane
    rmax : ndarray (nz,)
        The maximum allowed value for each plane

    Returns
    -------
    ndarray (nz, nt, ny, nx)
        The clipped movie
    """
    nz, __, __, __ = mov.shape
    for z in range(nz):
        mov[z, :, :, :].clip(rmin[z], rmax[z], out=mov[z, :, :, :])
    return mov


###Cpu registration function


def reg_3d_cpu(mov_frame, fft_3d_ref_conj, workers=-2):
    """
    fourier transform and multiply a 3D movie frame and the reference

    Parameters
    ----------
    mov_frame : ndarray (nz, ny, nx)
        A frame of the movie which needs to be registered
    fft_3d_ref_conj : ndarray (nz, ny, nx)
        The filterd fourrier transformed refference image
    workers : int, optional
        How many cpu cores to be used for the fourier transform, by default -2

    Returns
    -------
    ndarray (nz, ny, nx)
        The full phase_correlation for this frame
    """
    # Using scpiy fftn which is parallelised
    fft_3d_mov = scipy.fft.fftn(mov_frame[:, :, :], workers=workers)

    # numbs compiled and parallelised functions
    fft_3d_mov = div_norm_fft(fft_3d_mov)
    fft_correaltion = mult_fft(fft_3d_mov, fft_3d_ref_conj)

    phase_corr_frame = np.abs(scipy.fft.ifftn(fft_correaltion, workers=workers))
    return phase_corr_frame


def rigid_3d_ref_cpu(
    mov_cpu,
    mult_mask,
    add_mask,
    refs_f,
    pc_size,
    rmins=None,
    rmaxs=None,
    crosstalk_coeff=None,
    cavity_size=15,
):
    """
    Runs the 3d rigid registration for a 4D movie

    Parameters
    ----------
    mov_cpu : ndarray (nz, nt, ny, nx)
        The un-registered movie
    mult_mask : ndarray ( nz, ny, nx)
        The pre-calculated multiplcation mask
    add_mask : ndarray (nz, ny, nx)
        The pre-calcualted addition mask
    refs_f : nd array (nz, ny, nx)
        The filtered fourier transformed reference image
    pc_size : nd array (2*nz_pc + 1, 2*ny_pc + 1, 2*nx_pc + 1)
        The nQ_pc is the maximum shift allowed for the Q'th axis
    rmins : ndarray (nz), optional
        The minimum allowed value for each plane, by default None
    rmaxs : ndarray (nz), optional
        The maximum allowed value for each plane, by default None

    Returns
    -------
    phase_corr_shifted : ndarray (nt, 2*nz_pc +1, 2*ny_pc + 1, 2*nx_pc + 1)
        The phase correlation cropped and shifted so the peak is central
    shift : ndarray nt, (3)
        The integer shift to maximise phase correlation
    pc_peak_lock : ndarray (nt, 3)
        The index of the maximum value of the shift phase correlation array
    sub_pixel_shifts : ndarray (nt, 3)
        The sub pixel shift estiamted from the phase correlation
    """

    nz, nt, ny, nx = mov_cpu.shape
    max_pc_size = pc_size * 2 + 1
    phase_corr_shifted = np.zeros((nt, max_pc_size[0], max_pc_size[1], max_pc_size[2]))
    int_shift = np.zeros((nt, 3))
    pc_peak_loc = np.zeros((nt, 3))
    sub_pixel_shifts = np.zeros((nt, 3))

    if crosstalk_coeff is not None:
        mov_cpu = utils.crosstalk_subtract(mov_cpu, crosstalk_coeff, cavity_size)
    if np.logical_or(np.all(rmins != None), np.all(rmaxs != None)):
        mov_cpu = clip_mov_cpu(mov_cpu, rmins, rmaxs)

    masked_mov = np.zeros_like(mov_cpu)
    masked_mov = apply_mask4D(mov_cpu, mult_mask, add_mask, masked_mov)

    for t in range(nt):
        phase_corr_tmp = reg_3d_cpu(masked_mov[:, t, :, :], refs_f, workers=-1)
        phase_corr_shifted[t], int_shift[t], pc_peak_loc[t], sub_pixel_shifts[t] = (
            process_phase_corr_per_frame(phase_corr_tmp, pc_size)
        )

    return phase_corr_shifted, int_shift, pc_peak_loc, sub_pixel_shifts


## GPU registration function
def rigid_3d_ref_gpu(
    mov_cpu,
    mult_mask,
    add_mask,
    refs_f,
    pc_size,
    batch_size=20,
    rmins=None,
    rmaxs=None,
    crosstalk_coeff=None,
    shift_reg=False,
    xpad=None,
    ypad=None,
    fuse_shift=None,
    new_xs=None,
    old_xs=None,
    plane_shifts=None,
    process_mov=False,
    cavity_size=15,
):
    """
    Runs rigid registration on the gpu.

    Parameters
    ----------
    mov_cpu : ndarray (nz, nt, ny*, nx*)
        The un-registered movie, * may be un-fused
    mult_mask : ndarray ( nz, ny, nx)
        The pre-calculated multiplcation mask
    add_mask : ndarray (nz, ny, nx)
        The pre-calcualted addition mask
    refs_f : nd array (nz, ny, nx)
        The filtered fourier transformed reference image
    pc_size : nd array (2*nz_pc + 1, 2*ny_pc + 1, 2*nx_pc + 1)
        The nQ_pc is the maximum shift allowed for the Q'th axis
    rmins : ndarray (nz), optional
        The minimum allowed value for each plane, by default None
    rmaxs : ndarray (nz), optional
        The maximum allowed value for each plane, by default None
    crosstalk_coeff : float, optional
        The value of crosstalk across LBM cavities, by default None

    Returns
    -------
    phase_corr_shifted : ndarray (nt, 2*nz_pc +1, 2*ny_pc + 1, 2*nx_pc + 1)
        The phase correlation cropped and shifted so the peak is central
    shift : ndarray nt, (3)
        The integer shift to maximise phase correlation
    pc_peak_lock : ndarray (nt, 3)
        The index of the maximum value of the shift phase correlation array
    sub_pixel_shifts : ndarray (nt, 3)
        The sub pixel shift estiamted from the phase correlation
    """
    if not HAS_CUPY:
        raise ImportError(
            "GPU registration requires cupy. Please install cupy to use this function."
            " See https://docs.cupy.dev/en/stable/install.html for installation instructions."
        )
    mempool = cp.get_default_memory_pool()
    __, nt, __, __ = mov_cpu.shape
    max_pc_size = pc_size * 2 + 1

    phase_corr_shifted = np.zeros((nt, max_pc_size[0], max_pc_size[1], max_pc_size[2]))
    int_shift = np.zeros((nt, 3), dtype=np.int32)
    pc_peak_loc = np.zeros((nt, 3), dtype=np.int32)
    sub_pixel_shifts = np.zeros((nt, 3))
    mov_cpu_processed = None

    if shift_reg == True:
        mov_shifted = np.zeros_like(mov_cpu)
    total_batches = int(np.ceil(nt / batch_size))
    for b in range(total_batches):
        mempool.free_all_blocks()
        t1 = b * batch_size  # starting time point of batch
        t2 = int(np.min((nt, (b + 1) * batch_size)))  # end time point of batch

        if process_mov:
            mov_gpu = cp.asarray(mov_cpu[:, t1:t2, :, :])
            mov_gpu, mov_cpu_processed_tmp = process_mov_gpu(
                mov_gpu,
                plane_shifts,
                xpad,
                ypad,
                fuse_shift,
                new_xs,
                old_xs,
                crosstalk_coeff=crosstalk_coeff,
                cavity_size=cavity_size,
            )
            # ov_cpu_processed needs to be fused & padded but NOT spatially subseted!
            if mov_cpu_processed is None:
                # allocate CPU array for fused & padded movie ("processed")
                mov_cpu_processed = n.zeros(
                    (
                        mov_cpu_processed_tmp.shape[0],
                        nt,
                        mov_cpu_processed_tmp.shape[2],
                        mov_cpu_processed_tmp.shape[3],
                    ),
                    n.float32,
                )
            mov_cpu_processed[:, t1:t2] = mov_cpu_processed_tmp
        else:
            mov_gpu = cp.asarray(mov_cpu[:, t1:t2, :, :])
            if crosstalk_coeff is not None:
                mov_gpu = utils.crosstalk_subtract(mov_gpu, crosstalk_coeff, cavity_size)
        mult_mask = cp.asarray(mult_mask)
        add_mask = cp.asarray(add_mask)

        if np.logical_or(np.all(rmins != None), np.all(rmaxs != None)):
            mov_gpu = clip_mov_gpu(mov_gpu, rmins, rmaxs)

        mov_gpu = apply_mask4D_gpu(mov_gpu, mult_mask, add_mask)

        phase_corr_tmp = reg_3d_gpu(mov_gpu[:, :, :, :], refs_f)
        (
            phase_corr_shifted[t1:t2],
            int_shift[t1:t2],
            pc_peak_loc[t1:t2],
            sub_pixel_shifts[t1:t2],
        ) = process_phase_corr_gpu(phase_corr_tmp, cp.asarray(pc_size))

        if shift_reg == True:
            mov_gpu = shift_gpu(mov_gpu, int_shift[t1:t2])
            mov_shifted[:, t1:t2, :, :] = mov_gpu.get()

        del mov_gpu
        del phase_corr_tmp
        mempool.free_all_blocks()

    mempool.free_all_blocks()
    if shift_reg == True:
        return phase_corr_shifted, int_shift, pc_peak_loc, sub_pixel_shifts, mov_shifted
    else:
        return (
            phase_corr_shifted,
            int_shift,
            pc_peak_loc,
            sub_pixel_shifts,
            mov_cpu_processed,
        )


def rigid_3d_ref_gpu_dev(
    mov_cpu,
    mult_mask,
    add_mask,
    refs_f,
    pc_size,
    cropy,
    cropx,
    batch_size=20,
    rmins=None,
    rmaxs=None,
    crosstalk_coeff=None,
    shift_reg=False,
):
    """
    Runs rigid registration on the gpu.
    #NOTE this is set up to test, sending the full movie and croppping on the gpu.

    Parameters
    ----------
    mov_cpu : ndarray (nz, nt, ny, nx)
        The un-registered movie
    mult_mask : ndarray ( nz, ny, nx)
        The pre-calculated multiplcation mask
    add_mask : ndarray (nz, ny, nx)
        The pre-calcualted addition mask
    refs_f : nd array (nz, ny, nx)
        The filtered fourier transformed reference image
    pc_size : nd array (2*nz_pc + 1, 2*ny_pc + 1, 2*nx_pc + 1)
        The nQ_pc is the maximum shift allowed for the Q'th axis
    rmins : ndarray (nz), optional
        The minimum allowed value for each plane, by default None
    rmaxs : ndarray (nz), optional
        The maximum allowed value for each plane, by default None
    crosstalk_coeff : float, optional
        The value of crosstalk across LBM cavities, by default None

    Returns
    -------
    phase_corr_shifted : ndarray (nt, 2*nz_pc +1, 2*ny_pc + 1, 2*nx_pc + 1)
        The phase correlation cropped and shifted so the peak is central
    shift : ndarray nt, (3)
        The integer shift to maximise phase correlation
    pc_peak_lock : ndarray (nt, 3)
        The index of the maximum value of the shift phase correlation array
    sub_pixel_shifts : ndarray (nt, 3)
        The sub pixel shift estiamted from the phase correlation
    """
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    nz, nt, ny, nx = mov_cpu.shape
    max_pc_size = pc_size * 2 + 1

    phase_corr_shifted = np.zeros((nt, max_pc_size[0], max_pc_size[1], max_pc_size[2]))
    int_shift = np.zeros((nt, 3))
    pc_peak_loc = np.zeros((nt, 3))
    sub_pixel_shifts = np.zeros((nt, 3))

    if shift_reg == True:
        mov_shifted = np.zeros_like(mov_cpu)

    total_batches = int(np.ceil(nt / batch_size))
    for b in range(total_batches):
        t1 = b * batch_size  # starting time point of batch
        t2 = int(np.min((nt, (b + 1) * batch_size)))  # end time point of batch

        mov_gpu = cp.asarray(mov_cpu[:, t1:t2, :, :])
        mov_reg = cp.zeros(mov_gpu[:, :, cropy:-cropy, cropx:-cropx].shape)
        mov_reg = mov_gpu[:, :, cropy:-cropy, cropx:-cropx]

        mult_mask = cp.asarray(mult_mask)
        add_mask = cp.asarray(add_mask)

        if crosstalk_coeff is not None:
            mov_gpu = reg.crosstalk_subtract(mov_gpu, crosstalk_coeff)
        if np.logical_or(np.all(rmins != None), np.all(rmaxs != None)):
            mov_gpu = clip_mov_gpu(mov_gpu, rmins, rmaxs)

        mov_reg = apply_mask4D_gpu(mov_reg, mult_mask, add_mask)
        # NOTE not cropping mov here
        phase_corr_tmp = reg_3d_gpu(mov_reg, refs_f)
        (
            phase_corr_shifted[t1:t2],
            int_shift[t1:t2],
            pc_peak_loc[t1:t2],
            sub_pixel_shifts[t1:t2],
        ) = process_phase_corr_gpu(phase_corr_tmp, cp.asarray(pc_size))

        if shift_reg == True:
            mov_gpu = shift_gpu(mov_gpu, int_shift[t1:t2])
            mov_shifted[:, t1:t2, :, :] = mov_gpu.get()

        print(f"completed batch {b}")
        mempool.free_all_blocks()
    mempool.free_all_blocks()
    if shift_reg == True:
        return phase_corr_shifted, int_shift, pc_peak_loc, sub_pixel_shifts, mov_shifted
    else:
        return phase_corr_shifted, int_shift, pc_peak_loc, sub_pixel_shifts


def shift_gpu(mov_gpu, shift_batch):
    # for a batch

    __, ntb, __, __ = mov_gpu.shape
    for t in range(ntb):
        mov_gpu[:, t, :, :] = cp.roll(
            mov_gpu[:, t, :, :], (shift_batch[t, 1], shift_batch[t, 2]), axis=(1, 2)
        )
        if shift_batch[t, 2] > 0:
            mov_gpu[:, t, :, : shift_batch[t, 2]] = 0
        elif shift_batch[t, 2] < 0:
            mov_gpu[:, shift_batch[t, 2] :] = 0
        if shift_batch[t, 1] > 0:
            mov_gpu[:, t:, : shift_batch[t, 1], :] = 0
        elif shift_batch[t, 1] < 0:
            mov_gpu[:, t, shift_batch[t, 1] :, :] = 0

    return mov_gpu


def clip_mov_gpu(mov, rmin, rmax):
    nz, __, __, __ = mov.shape
    for z in range(nz):
        mov[z, :, :, :].real = cp.clip(mov[z, :, :, :].real, rmin[z], rmax[z])
    return mov


def apply_mask4D_gpu(data, mask_mul, mask_offset):
    for t in range(data.shape[1]):
        data[:, t, :, :] = data[:, t, :, :] * mask_mul + mask_offset
    return data


def process_mov_gpu(
    mov_gpu,
    plane_shifts,
    xpad,
    ypad,
    fuse_shift,
    new_xs,
    old_xs,
    crosstalk_coeff=None,
    cavity_size=15,
):
    # fuse and pad the movie
    # TODO solve the xpad ypad integer vs array conflict
    mov_gpu = fuse_and_pad_gpu(
        mov_gpu, fuse_shift, np.array(ypad), np.array(xpad), new_xs, old_xs
    )
    mov_gpu = mov_gpu.real
    # subtract crosstalk between cavities if given, BEFORE plane shifts
    if crosstalk_coeff is not None:
        mov_gpu = utils.crosstalk_subtract(mov_gpu, crosstalk_coeff, cavity_size)
    # apply the lbm shifts
    mov_gpu = shift_mov_lbm_gpu(mov_gpu, plane_shifts)
    # get the processed movie on the cpu
    mov_cpu_processed_tmp = mov_gpu.get()
    # crop the movie so only full z-planes count
    if xpad > 0:
        mov_gpu = mov_gpu[:, :, :, xpad:-xpad]
    if ypad > 0:
        mov_gpu = mov_gpu[:, :, ypad:-ypad, :]
    return mov_gpu, mov_cpu_processed_tmp


# TODO changed from register_gpu, 1. pads became int
# 2 added the shift to x-axis so the blanck space is on the left side
def fuse_and_pad_gpu(mov_gpu, fuse_shift, ypad, xpad, new_xs, old_xs):
    nz, nt, ny, nx = mov_gpu.shape
    n_stitches = len(new_xs) - 1
    n_xpix_lost_fusing = n_stitches * fuse_shift
    nyn = ny + ypad
    nxn = nx + xpad - n_xpix_lost_fusing

    mov_pad = cp.zeros((nz, nt, nyn, nxn), dtype=cp.complex64)
    for strip_idx in range(len(new_xs)):
        nx0, nx1 = new_xs[strip_idx]
        ox0, ox1 = old_xs[strip_idx]
        mov_pad[:, :, :ny, xpad + nx0 : xpad + nx1] = mov_gpu[:, :, :, ox0:ox1]

    return mov_pad


def shift_mov_lbm_gpu(mov_gpu, plane_shifts, fill_value=0):
    """
    Apply LBM shifts over a 4D movie, these shifts are same for all time and have different x/y shifts per z-plane

    Parameters
    ----------
    mov : nd array (nz, nt, ny, nx)
        Movie to be shifted
    plane_shifts : ndarray ( nz, 2)
        The (y_shift, x_shift) per z-plane to allign the planes
    fill_value : optional
        The vaule to fill the empty spaces caused by shifting the array, by default 0

    Returns
    -------
    ndarray (nz, nt, ny, nx)
        The shifted array
    """

    nz, __, __, __ = mov_gpu.shape

    for z in range(nz):
        shift = plane_shifts[z, :]
        # print(shift)
        if (shift[0] == 0) & (shift[1] == 0):  # 00
            mov_gpu[z, :, :, :] = mov_gpu[z, :, :, :]
        elif (shift[0] > 0) & (shift[1] > 0):  # ++
            mov_gpu[z, :, shift[0] :, shift[1] :] = mov_gpu[
                z, :, : -shift[0], : -shift[1]
            ]
            mov_gpu[z, :, :, : shift[1]] = fill_value
            mov_gpu[z, :, : shift[0], :] = fill_value
        elif (shift[0] > 0) & (shift[1] < 0):  # +-
            mov_gpu[z, :, shift[0] :, : shift[1]] = mov_gpu[
                z, :, : -shift[0], -shift[1] :
            ]
            mov_gpu[z, :, :, shift[1] :] = fill_value
            mov_gpu[z, :, : shift[0], :] = fill_value
        elif (shift[0] == 0) & (shift[1] < 0):  # 0-
            mov_gpu[z, :, shift[0] :, : shift[1]] = mov_gpu[z, :, :, -shift[1] :]
            mov_gpu[z, :, :, shift[1] :] = fill_value
        elif (shift[0] > 0) & (shift[1] == 0):  # +0
            mov_gpu[z, :, shift[0] :, :] = mov_gpu[z, :, : -shift[0], :]
            mov_gpu[z, :, : shift[0], :] = fill_value

        elif (shift[0] < 0) & (shift[1] > 0):  # -+
            mov_gpu[z, :, : shift[0], shift[1] :] = mov_gpu[
                z, :, -shift[0] :, : -shift[1]
            ]
            mov_gpu[z, :, :, : shift[1]] = fill_value
            mov_gpu[z, :, shift[0] :, :] = fill_value

        elif (shift[0] < 0) & (shift[1] == 0):  # -0
            mov_gpu[z, :, : shift[0], :] = mov_gpu[z, :, -shift[0] :, :]
            mov_gpu[z, :, shift[0] :, :] = fill_value
        elif (shift[0] == 0) & (shift[1] > 0):  # 0+
            mov_gpu[z, :, :, shift[1] :] = mov_gpu[z, :, :, : -shift[1]]
            mov_gpu[z, :, :, : shift[1]] = fill_value

        else:  # --
            mov_gpu[z, :, : shift[0], : shift[1]] = mov_gpu[
                z, :, -shift[0] :, -shift[1] :
            ]
            mov_gpu[z, :, :, shift[1] :] = fill_value
            mov_gpu[z, :, shift[0] :, :] = fill_value
    return mov_gpu


# decide when/where to calc/get masks + fft'd filterd ref img
def reg_3d_gpu(mov_batch_gpu, fft_3d_ref_conj):
    """
    fourier transform and multiply a 3D movie batch and the reference, ran on the GPU

    Parameters
    ----------
    mov_batch_gpu : ndarray (nz, nt_batch, ny, nx)
        A batch of the movie which needs to be registered, on the gpu
    fft_3d_ref_conj : ndarray (nz, ny, nx)
        The filterd fourrier transformed refference image

    Returns
    -------
    ndarray (nz,nt_batch, ny, nx)
        The full phase_correlation for this frame
    """
    nz, nt, ny, nx = mov_batch_gpu.shape

    fft_3d_ref_conj_gpu = cp.asarray(fft_3d_ref_conj)
    phase_corr_batch = cp.zeros((nt, nz, ny, nx), dtype=cp.float64)
    # Using  cpy fftn
    fft_3d_mov = cufft.fftn(mov_batch_gpu[:, :, :, :], axes=(0, 2, 3))

    for t in range(nt):
        fft_3d_mov[:, t, :, :] = fft_3d_mov[:, t, :, :] / (
            1e-5 + cp.abs(fft_3d_mov[:, t, :, :])
        )
        fft_3d_mov[:, t, :, :] = fft_3d_mov[:, t, :, :] * fft_3d_ref_conj_gpu

    phase_corr_batch = cp.abs(cufft.ifftn(fft_3d_mov, axes=(0, 2, 3))).swapaxes(0, 1)

    del fft_3d_mov

    return phase_corr_batch


def process_phase_corr_gpu(phase_corr, pc_size):
    """
    Analysise the phase correlation to return useful information, a re-aranged phase_corr, peak location and
    integer + sub pixel shifts.
    This function is used for the gpu where registration is done in batches.

    Parameters
    ----------
    phase_corr : ndarray (nz, ny, nx)
        The full phase correlation for a frame
    pc_size : ndarray (nz_pc, ny_pc, nx_pc)
        This determines the size of the re-aranged phase correlation array and the maximum size of shifts allowed

    Returns
    -------
    phase_corr_shifted : ndarray (2*nz_pc +1, 2*ny_pc + 1, 2*nx_pc + 1)
        The phase correlation cropped and shifted so the peak is central
    shift : ndarray (3,)
        The integer shift to maximise phase correlation
    pc_peak_lock : ndarray (3,)
        The index of the maximum value of the shift phase correlation array
    sub_pixel_shifts : ndarray (3,)
        The sub pixel shift estiamted from the phase correlation

    """
    max_pc_size = pc_size * 2 + 1
    nt, nz, ny, nx = phase_corr.shape
    phase_corr_shifted = cp.zeros(
        (nt, int(max_pc_size[0]), int(max_pc_size[1]), int(max_pc_size[2])),
        dtype=cp.float64,
    )

    # want z planes 0,1,2 to go to 2,3,4
    # want z planes 14,13 to go to 1,0
    # so the new z plane -2 is the shift!
    # asfor x/y 0-50 goes to 50-101
    # and the last 50 go to 0-50

    # have z+/- x+/- y+/-
    # add z+ x+ y+
    phase_corr_shifted[:, pc_size[0] :, pc_size[1] :, pc_size[2] :] = phase_corr[
        :, : pc_size[0] + 1, : pc_size[1] + 1, : pc_size[2] + 1
    ]
    # add z+ x- y+
    phase_corr_shifted[:, pc_size[0] :, pc_size[1] :, : pc_size[2]] = phase_corr[
        :, : pc_size[0] + 1, : pc_size[1] + 1, nx - pc_size[2] :
    ]
    # add z+ x+ y-
    phase_corr_shifted[:, pc_size[0] :, : pc_size[1], pc_size[2] :] = phase_corr[
        :, : pc_size[0] + 1, ny - pc_size[1] :, : pc_size[2] + 1
    ]
    # add z+ x- y-
    phase_corr_shifted[:, pc_size[0] :, : pc_size[1], : pc_size[2]] = phase_corr[
        :, : pc_size[0] + 1, ny - pc_size[1] :, nx - pc_size[2] :
    ]

    # add z- x+ y+
    phase_corr_shifted[:, : pc_size[0], pc_size[1] :, pc_size[2] :] = phase_corr[
        :, nz - pc_size[0] :, : pc_size[1] + 1, : pc_size[2] + 1
    ]
    # add z- x- y+
    phase_corr_shifted[:, : pc_size[0], pc_size[1] :, : pc_size[2]] = phase_corr[
        :, nz - pc_size[0] :, : pc_size[1] + 1, nx - pc_size[2] :
    ]
    # add z- x+ y-
    phase_corr_shifted[:, : pc_size[0], : pc_size[1], pc_size[2] :] = phase_corr[
        :, nz - pc_size[0] :, ny - pc_size[1] :, : pc_size[2] + 1
    ]
    # add z- x- y-
    phase_corr_shifted[:, : pc_size[0], : pc_size[1], : pc_size[2]] = phase_corr[
        :, nz - pc_size[0] :, ny - pc_size[1] :, nx - pc_size[2] :
    ]

    # SWITCHING back to cpu here
    phase_corr_shifted = cp.asnumpy(phase_corr_shifted)
    pc_size = cp.asnumpy(pc_size)

    shift = np.zeros((nt, 3))
    pc_peak_loc = np.zeros((nt, 3), dtype=np.int16)
    shape = phase_corr_shifted[0].shape
    for t in range(nt):
        mx = np.argmax(phase_corr_shifted[t])
        pc_peak_loc[t, :] = np.unravel_index(
            mx, shape
        )  # cp.asarray(cp.unravel_index(mx, shape))
        shift[t, :] = pc_peak_loc[t, :] - pc_size

    z_sub_pixel = cp.zeros(nt)
    y_sub_pixel = cp.zeros(nt)
    x_sub_pixel = cp.zeros(nt)
    for t in range(nt):
        z_sub_pixel[t] = est_sub_pixel_shift(
            phase_corr_shifted[t, :, pc_peak_loc[t, 1], pc_peak_loc[t, 2]]
        )
        y_sub_pixel[t] = est_sub_pixel_shift(
            phase_corr_shifted[t, pc_peak_loc[t, 0], :, pc_peak_loc[t, 2]]
        )
        x_sub_pixel[t] = est_sub_pixel_shift(
            phase_corr_shifted[t, pc_peak_loc[t, 0], pc_peak_loc[t, 1], :]
        )

    # Somehow the result of np.vstack is a cp.ndarray so need to do cp.asnumpy, this is true try tofind out why?
    sub_pixel_shifts = cp.asnumpy(np.vstack([z_sub_pixel, y_sub_pixel, x_sub_pixel]).T)

    return phase_corr_shifted, shift, pc_peak_loc, sub_pixel_shifts


@njit(parallel=True, nogil=True, cache=True)
def shift_mov_fast(mov, shifts, fill_value=0):
    """
    A fast function, which applies x/y integer shifts

    Parameters
    ----------
    mov : ndarray (nz, nt, ny, nx)
        Movie to be shifted
    shifts : ndarray (nt, 3)
        shifts have (z_shift, y_shift, x_shift) for each frame, doesnt apply z-shifts as should be 0!!
        + y is shifting 'right, + x is shifting 'down'
    fill_value : float, optional
        the value to fill the overlaped empty voxels, by default 0

    Returns
    -------
    ndarray
        The shifted movie
    """
    shifted_mov = np.zeros_like(mov)
    __, nt, __, __ = mov.shape
    for t in range(nt):
        # print(f"FAst mov shift {t}/{nt}")
        shift = shifts[t, :]

        # add the 4 cases where one of the shifts is0
        if (shift[1] > 0) & (shift[2] > 0):  # ++
            shifted_mov[:, t, :, : shift[2]] = fill_value
            shifted_mov[:, t, : shift[1], :] = fill_value
            shifted_mov[:, t, shift[1] :, shift[2] :] = mov[
                :, t, : -shift[1], : -shift[2]
            ]
        elif (shift[1] == 0) & (shift[2] == 0):  # 00
            shifted_mov[:, t, :, :] = mov[:, t, :, :]
        elif (shift[1] == 0) & (shift[2] > 0):  # 0+
            shifted_mov[:, t, :, : shift[2]] = fill_value
            shifted_mov[:, t, :, shift[2] :] = mov[:, t, :, : -shift[2]]
        elif (shift[1] == 0) & (shift[2] < 0):  # 0-
            shifted_mov[:, t, :, shift[2] :] = fill_value
            shifted_mov[:, t, :, : shift[2]] = mov[:, t, :, -shift[2] :]
        elif (shift[1] > 0) & (shift[2] == 0):  # +0
            shifted_mov[:, t, : shift[1], :] = fill_value
            shifted_mov[:, t, shift[1] :, :] = mov[:, t, : -shift[1], :]
        elif (shift[1] < 0) & (shift[2] == 0):  # -0
            shifted_mov[:, t, shift[1] :, :] = fill_value
            shifted_mov[:, t, : shift[1], :] = mov[:, t, -shift[1] :, :]
        elif (shift[1] > 0) & (shift[2] < 0):  # +-
            shifted_mov[:, t, :, shift[2] :] = fill_value
            shifted_mov[:, t, : shift[1], :] = fill_value
            shifted_mov[:, t, shift[1] :, : shift[2]] = mov[
                :, t, : -shift[1], -shift[2] :
            ]
        elif (shift[1] < 0) & (shift[2] > 0):  # -+
            shifted_mov[:, t, :, : shift[2]] = fill_value
            shifted_mov[:, t, shift[1] :, :] = fill_value
            shifted_mov[:, t, : shift[1], shift[2] :] = mov[
                :, t, -shift[1] :, : -shift[2]
            ]
        else:  # --
            shifted_mov[:, t, :, shift[2] :] = fill_value
            shifted_mov[:, t, shift[1] :, :] = fill_value
            shifted_mov[:, t, : shift[1], : shift[2]] = mov[
                :, t, -shift[1] :, -shift[2] :
            ]

    return shifted_mov

@njit(parallel=True, nogil=True, cache=True)
def shift_mov_z(mov, shifts, fill_value=0):
    """
    A fast function, which applies z integer shifts

    Parameters
    ----------
    mov : ndarray (nz, nt, ny, nx)
        Movie to be shifted
    shifts : ndarray (nt, 3)
        shifts have (z_shift, y_shift, x_shift) for each frame
    fill_value : float, optional
        the value to fill the overlaped empty voxels, by default 0

    Returns
    -------
    ndarray
        The shifted movie
    """
    shifted_mov = np.zeros_like(mov)
    __, nt, __, __ = mov.shape
    for t in range(nt):
        shift = shifts[t, :]

        if shift[0] > 0:  # +
            shifted_mov[:shift[0], t, :, :] = fill_value
            shifted_mov[shift[0]:, t, :, :] = mov[:-shift[0], t, :, :]
        if shift[0] < 0:  # -
            shifted_mov[shift[0]:, t, :, :] = fill_value
            shifted_mov[:shift[0], t, :, :] = mov[-shift[0]:, t, :, :]

    return shifted_mov

# dev function for finding translation between two 3d images
def register_2_images(img1, img2, pc_size):
    """
    Will use phase correlation registration to eastimate the shift between two different images.
    The 2 images need to be the same size, one img can be padded e.g:
        empty2 = np.zeros_like(img1)
        nz, ny, nx = img2.shape
        empty2[:, :ny, :nx] = ref_img_diff_ref
    above only works if img 2 is smaller in both axis, need to pad both or pad and crop if one axis is bigger and the other is smaller

    Parameters
    ----------
    img1 : ndarray (nz, ny, nx)
        The first img
    img2 : ndarray (nz, ny, nx)
        The second img
    pc_size : ndarray
        (z_crop_size, y_crop_size, x_crop_size)
    """
    fft_img1 = scipy.fft.fftn(img1, workers=-1)
    fft_img2 = scipy.fft.fftn(img2, workers=-1)

    fft_img2_conj = np.conj(fft_img2)

    fft_product = fft_img1 * fft_img2_conj / (np.abs(fft_img1 * fft_img2_conj))
    phase_corr = np.abs(scipy.fft.ifftn(fft_product))

    phase_corr_shifted, shift, pc_peak_loc, sub_pixel_shifts = (
        process_phase_corr_per_frame(phase_corr, pc_size)
    )

    return phase_corr_shifted, shift, pc_peak_loc, sub_pixel_shifts
