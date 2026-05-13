# new function used for the 3d registration
import os
from functools import lru_cache
import time
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
from .utils import default_log


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


def kernelD3(xs: n.ndarray, ys: n.ndarray, zs: n.ndarray):
    """
    3D gaussian kernel for nonrigid smoothing across blocks.

    Parameters
    ----------
    xs, ys, zs : ndarray
        1D arrays of block indices along x, y, z.

    Returns
    -------
    ndarray
        (nb, nb) smoothing matrix, columns normalized.
    """
    zz, yy, xx = n.meshgrid(zs, ys, xs, indexing="ij")
    coords = n.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
    diffs = coords[:, None, :] - coords[None, :, :]
    dist2 = (diffs**2).sum(axis=2)
    R = n.exp(-dist2)
    R = R / n.sum(R, axis=0)
    return R


def make_blocks_3d(Lz, Ly, Lx, block_size=(8, 128, 128)):
    """
    Computes overlapping blocks to split a 3D volume for nonrigid registration.

    Parameters
    ----------
    Lz, Ly, Lx : int
        Volume dimensions (z, y, x)
    block_size : tuple
        Desired block size (bz, by, bx)

    Returns
    -------
    zblock, yblock, xblock : list
        Lists of [start, end] index arrays per block.
    nblocks : list
        [nz, ny, nx] number of blocks per dimension.
    block_size : tuple
        Actual block size used per dimension.
    nonrigid_smoothing_matrix : ndarray
        (nb, nb) block-smoothing matrix.
    """
    block_size_z, nz = ref.calculate_nblocks(N=Lz, block_size=block_size[0])
    block_size_y, ny = ref.calculate_nblocks(N=Ly, block_size=block_size[1])
    block_size_x, nx = ref.calculate_nblocks(N=Lx, block_size=block_size[2])
    block_size = (block_size_z, block_size_y, block_size_x)

    def _unique_linspace_starts(start, stop, npts):
        if npts <= 1 or stop < start:
            return n.array([int(start)])
        starts = n.linspace(start, stop, npts)
        starts = n.rint(starts).astype(int)
        starts = n.unique(starts)
        if starts.size > 0 and starts[-1] != int(stop):
            if int(stop) not in starts:
                starts = n.append(starts, int(stop))
        return starts

    zstart = _unique_linspace_starts(0, Lz - block_size[0], nz)
    ystart = _unique_linspace_starts(0, Ly - block_size[1], ny)
    xstart = _unique_linspace_starts(0, Lx - block_size[2], nx)

    nz = zstart.shape[0]
    ny = ystart.shape[0]
    nx = xstart.shape[0]

    zblock = [
        n.array([zstart[iz], zstart[iz] + block_size[0]])
        for iz in range(nz)
        for _ in range(ny)
        for _ in range(nx)
    ]
    yblock = [
        n.array([ystart[iy], ystart[iy] + block_size[1]])
        for _ in range(nz)
        for iy in range(ny)
        for _ in range(nx)
    ]
    xblock = [
        n.array([xstart[ix], xstart[ix] + block_size[2]])
        for _ in range(nz)
        for _ in range(ny)
        for ix in range(nx)
    ]

    nonrigid_smoothing_matrix = kernelD3(
        xs=n.arange(nx), ys=n.arange(ny), zs=n.arange(nz)
    ).T

    return zblock, yblock, xblock, [nz, ny, nx], block_size, nonrigid_smoothing_matrix


def _block_centers_3d(zblocks, yblocks, xblocks):
    zstarts = n.array([zb[0] for zb in zblocks], dtype=n.float32)
    ystarts = n.array([yb[0] for yb in yblocks], dtype=n.float32)
    xstarts = n.array([xb[0] for xb in xblocks], dtype=n.float32)

    zuniq = n.unique(zstarts)
    yuniq = n.unique(ystarts)
    xuniq = n.unique(xstarts)

    bz = float(zblocks[0][1] - zblocks[0][0])
    by = float(yblocks[0][1] - yblocks[0][0])
    bx = float(xblocks[0][1] - xblocks[0][0])

    zcenters = zuniq + bz * 0.5
    ycenters = yuniq + by * 0.5
    xcenters = xuniq + bx * 0.5

    return zcenters, ycenters, xcenters


def _make_block_index_grid_3d(Lz, Ly, Lx, zcenters, ycenters, xcenters):
    zcenters = cp.asarray(zcenters, dtype=cp.float32)
    ycenters = cp.asarray(ycenters, dtype=cp.float32)
    xcenters = cp.asarray(xcenters, dtype=cp.float32)

    nzb = zcenters.shape[0]
    nyb = ycenters.shape[0]
    nxb = xcenters.shape[0]

    # When one block exists per voxel along an axis (bz=1 / by=1 / bx=1),
    # use the identity mapping so each voxel reads its own block's shift.
    # Otherwise linearly interpolate between block centers so the per-voxel
    # warp is C0-smooth across block boundaries.
    if nzb == Lz:
        z_idx = cp.arange(Lz, dtype=cp.float32)
    else:
        z_idx = cp.interp(
            cp.arange(Lz, dtype=cp.float32),
            zcenters,
            cp.arange(nzb, dtype=cp.float32),
        )
    if nyb == Ly:
        y_idx = cp.arange(Ly, dtype=cp.float32)
    else:
        y_idx = cp.interp(
            cp.arange(Ly, dtype=cp.float32),
            ycenters,
            cp.arange(nyb, dtype=cp.float32),
        )
    if nxb == Lx:
        x_idx = cp.arange(Lx, dtype=cp.float32)
    else:
        x_idx = cp.interp(
            cp.arange(Lx, dtype=cp.float32),
            xcenters,
            cp.arange(nxb, dtype=cp.float32),
        )

    zq, yq, xq = cp.meshgrid(z_idx, y_idx, x_idx, indexing="ij")
    return zq, yq, xq


def nonrigid_transform_data_3d_gpu(
    mov_cpu,
    zshifts,
    yshifts,
    xshifts,
    zblocks,
    yblocks,
    xblocks,
    batch_size=5,
    order=1,
    mode="nearest",
    log_cb=default_log,
):
    """
    Apply 3D nonrigid block shifts to a movie on GPU using trilinear interpolation.

    Parameters
    ----------
    mov_cpu : ndarray (nz, nt, ny, nx)
        Movie to correct (CPU)
    zshifts, yshifts, xshifts : ndarray (nt, nzb, nyb, nxb)
        Block-wise 3D shifts (GPU or CPU)
    zblocks, yblocks, xblocks : list
        Block boundary arrays
    batch_size : int
        Time batch size for GPU processing
    order : int
        Interpolation order for map_coordinates (1 = linear)
    mode : str
        Boundary mode for map_coordinates

    Returns
    -------
    mov_corrected : ndarray (nz, nt, ny, nx)
        Nonrigid-corrected movie (CPU)
    """
    if not HAS_CUPY:
        raise ImportError(
            "GPU nonrigid correction requires cupy. Please install cupy to use this function."
        )

    nz, nt, ny, nx = mov_cpu.shape
    zcenters, ycenters, xcenters = _block_centers_3d(zblocks, yblocks, xblocks)
    zq, yq, xq = _make_block_index_grid_3d(nz, ny, nx, zcenters, ycenters, xcenters)

    zz, yy, xx = cp.meshgrid(
        cp.arange(nz, dtype=cp.float32),
        cp.arange(ny, dtype=cp.float32),
        cp.arange(nx, dtype=cp.float32),
        indexing="ij",
    )

    mov_gpu = cp.asarray(mov_cpu, dtype=cp.float32)
    mov_gpu = mov_gpu.swapaxes(0, 1)

    zshifts = cp.asarray(zshifts, dtype=cp.float32)
    yshifts = cp.asarray(yshifts, dtype=cp.float32)
    xshifts = cp.asarray(xshifts, dtype=cp.float32)

    mov_out = cp.zeros_like(mov_gpu)

    total_batches = int(n.ceil(nt / batch_size))
    for b in range(total_batches):
        t0 = b * batch_size
        t1 = int(n.min((nt, (b + 1) * batch_size)))

        block_coords = cp.stack([zq, yq, xq])
        for tid in range(t0, t1):
            zup = cuimage.map_coordinates(
                zshifts[tid], block_coords, order=order, mode="nearest"
            )
            yup = cuimage.map_coordinates(
                yshifts[tid], block_coords, order=order, mode="nearest"
            )
            xup = cuimage.map_coordinates(
                xshifts[tid], block_coords, order=order, mode="nearest"
            )

            coords = cp.stack([zz + zup, yy + yup, xx + xup])
            mov_out[tid] = cuimage.map_coordinates(
                mov_gpu[tid], coords, order=order, mode=mode, cval=0.0
            )

    mov_out = mov_out.swapaxes(0, 1)
    return mov_out.get()


def nonrigid_phasecorr_reference_3D(
    refImg0, maskSlope, smooth_sigma, zblock, yblock, xblock, sigz=None,
    voxel_size_um=None,
):
    """
    Computes taper masks and FFT'ed references for 3D nonrigid phase correlation.

    Parameters
    ----------
    refImg0 : ndarray
        Reference 3D volume (nz, ny, nx)
    maskSlope : float
        Spatial taper width for the full-volume mask
    smooth_sigma : float
        Gaussian filter width
    zblock, yblock, xblock : list
        Block boundary arrays
    sigz : float, optional
        Spatial taper width for z (defaults to maskSlope)
    voxel_size_um : tuple, optional
        (z, y, x) voxel size in microns. If provided, the per-block taper's
        z-sigma is scaled by `voxel_size_um[1] / voxel_size_um[0]` so the
        physical taper width matches between xy and z. With no voxel-size
        info available, falls back to the legacy isotropic-in-voxels behaviour
        (`block_sigz = 2 * smooth_sigma`).

    Returns
    -------
    maskMul : ndarray
        (nb, bz, by, bx) multiplication masks per block
    maskOffset : ndarray
        (nb, bz, by, bx) offset masks per block
    cfRefImg : ndarray
        (nb, bz, by, bx) FFT-domain references per block
    """
    nb = len(zblock)
    bz = zblock[0][1] - zblock[0][0]
    by = yblock[0][1] - yblock[0][0]
    bx = xblock[0][1] - xblock[0][0]

    if sigz is None:
        sigz = maskSlope

    # Per-block taper: xy sigma in voxels, z sigma scaled by the voxel
    # aspect ratio so the physical taper width matches across axes.
    block_sig_xy = 2 * smooth_sigma
    if voxel_size_um is not None:
        vz, vy, vx = float(voxel_size_um[0]), float(voxel_size_um[1]), float(voxel_size_um[2])
        block_sigz = block_sig_xy * (0.5 * (vy + vx) / vz)
    else:
        block_sigz = block_sig_xy

    gaussian_filter = gaussian_fft3D(smooth_sigma, bz, by, bx)

    maskMul = ref.spatial_taper3D(maskSlope, sigz, *refImg0.shape)
    maskMul1 = n.empty((nb, bz, by, bx), "float32")
    maskMul1[:] = ref.spatial_taper3D(block_sig_xy, block_sigz, bz, by, bx)
    maskOffset1 = n.empty((nb, bz, by, bx), "float32")
    cfRefImg1 = n.empty((nb, bz, by, bx), "complex64")
    refImg1 = n.empty((nb, bz, by, bx), "float32")

    for zind, yind, xind, maskMul1_n, maskOffset1_n, cfRefImg1_n, refImg1_n in zip(
        zblock, yblock, xblock, maskMul1, maskOffset1, cfRefImg1, refImg1
    ):
        ix = n.ix_(
            n.arange(zind[0], zind[-1]).astype("int"),
            n.arange(yind[0], yind[-1]).astype("int"),
            n.arange(xind[0], xind[-1]).astype("int"),
        )
        refImg = refImg0[ix]
        refImg1_n[:] = refImg

        # block-local masks, scaled by the global spatial taper
        maskMul1_n *= maskMul[ix]
        maskOffset1_n[:] = refImg.mean() * (1.0 - maskMul1_n)

        # gaussian filter in FFT domain
        cfRefImg1_n[:] = n.conj(scipy.fft.fftn(refImg))
        cfRefImg1_n /= 1e-5 + n.absolute(cfRefImg1_n)
        cfRefImg1_n[:] *= gaussian_filter

    return maskMul1, maskOffset1, cfRefImg1, refImg1


def get_nonrigid_phasecorr_and_masks_3d(ref_image, reference_params):
    """
    Produces the FFT'd 3D nonrigid references and masks.

    Parameters
    ----------
    ref_image : ndarray (nz, ny, nx)
        Reference 3D volume
    reference_params : dict
        Reference params containing block size and smoothing params

    Returns
    -------
    mult_mask_nr : ndarray
    add_mask_nr : ndarray
    refs_nr_f : ndarray
    refs_nr : ndarray
    zblock, yblock, xblock : list
    nonrigid_smoothing_matrix : ndarray
    reference_params : dict
    """
    nz, ny, nx = ref_image.shape
    smooth_sigma = reference_params["smooth_sigma_nr"]
    block_size = reference_params.get("block_size_3d", reference_params.get("block_size"))
    if block_size is None or len(block_size) != 3:
        raise ValueError("block_size_3d must be a 3-tuple for 3D nonrigid registration")

    (
        zblock,
        yblock,
        xblock,
        nblocks,
        block_size,
        nonrigid_smoothing_matrix,
    ) = make_blocks_3d(Lz=nz, Ly=ny, Lx=nx, block_size=block_size)

    reference_params["zblock"] = zblock
    reference_params["yblock"] = yblock
    reference_params["xblock"] = xblock
    reference_params["nblocks_3d"] = nblocks
    reference_params["block_size_3d"] = block_size
    reference_params["nonrigid_smoothing_matrix"] = nonrigid_smoothing_matrix

    mult_mask_nr, add_mask_nr, refs_nr_f, refs_nr = nonrigid_phasecorr_reference_3D(
        refImg0=ref_image,
        maskSlope=smooth_sigma * 3,
        smooth_sigma=smooth_sigma,
        zblock=zblock,
        yblock=yblock,
        xblock=xblock,
        voxel_size_um=reference_params.get("voxel_size_um", None),
    )

    return (
        mult_mask_nr,
        add_mask_nr,
        refs_nr_f,
        refs_nr,
        zblock,
        yblock,
        xblock,
        nonrigid_smoothing_matrix,
        reference_params,
    )


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


def spatial_autocorrelation_3d(
    vol,
    normalize=True,
    mean_subtract=True,
    use_gpu=None,
    boxsize=None,
    clip_percentiles=None,
):
    """
    Compute the 3D spatial autocorrelation of a volume over z, y, x.

    Parameters
    ----------
    vol : ndarray
        Input 3D volume (nz, ny, nx).
    normalize : bool
        If True, normalize by the maximum value.
    mean_subtract : bool
        If True, subtract the mean before autocorrelation.
    use_gpu : bool or None
        Force GPU (True) or CPU (False). If None, use GPU when vol is a cupy array.
    boxsize : tuple or None
        If provided, return a centered box of size (bz, by, bx) with the peak at
        (bz//2, by//2, bx//2). If length-2, interpreted as (bz, bxy) and returns
        a (bz, bxy) array where each z-slice is the average of the centered X and
        Y autocorrelation line profiles.
    clip_percentiles : tuple or None
        If provided, clip a copy of the volume to (low, high) percentiles before
        autocorrelation.

    Returns
    -------
    ndarray
        Autocorrelation volume, centered with fftshift.
    """
    if use_gpu and not HAS_CUPY:
        raise ImportError(
            "GPU autocorrelation requires cupy. Please install cupy to use this function."
        )

    xp = cp if (use_gpu or (HAS_CUPY and isinstance(vol, cp.ndarray))) else np
    vol_x = xp.asarray(vol).copy()
    if clip_percentiles is not None:
        p_low, p_high = clip_percentiles
        lo = xp.percentile(vol_x, p_low)
        hi = xp.percentile(vol_x, p_high)
        vol_x = xp.clip(vol_x, lo, hi)
    if mean_subtract:
        vol_x = vol_x - vol_x.mean()

    fft_vol = xp.fft.fftn(vol_x)
    ac = xp.fft.ifftn(fft_vol * xp.conj(fft_vol)).real
    ac = xp.fft.fftshift(ac)

    if boxsize is not None:
        if len(boxsize) == 2:
            bz, bxy = (int(boxsize[0]), int(boxsize[1]))
            cz, cy, cx = (ac.shape[0] // 2, ac.shape[1] // 2, ac.shape[2] // 2)
            z0 = cz - bz // 2
            z1 = z0 + bz
            y0 = cy - bxy // 2
            x0 = cx - bxy // 2
            y1 = y0 + bxy
            x1 = x0 + bxy
            if (
                z0 < 0
                or y0 < 0
                or x0 < 0
                or z1 > ac.shape[0]
                or y1 > ac.shape[1]
                or x1 > ac.shape[2]
            ):
                raise ValueError("boxsize exceeds autocorrelation volume dimensions")

            out = xp.zeros((bz, bxy), dtype=ac.dtype)
            for zi, zidx in enumerate(range(z0, z1)):
                line_x = ac[zidx, cy, x0:x1]
                line_y = ac[zidx, y0:y1, cx]
                out[zi] = 0.5 * (line_x + line_y)
            if normalize:
                out = out / (out.max() + 1e-8)
            return out

        bz, by, bx = (int(boxsize[0]), int(boxsize[1]), int(boxsize[2]))
        cz, cy, cx = (ac.shape[0] // 2, ac.shape[1] // 2, ac.shape[2] // 2)
        z0 = cz - bz // 2
        y0 = cy - by // 2
        x0 = cx - bx // 2
        z1 = z0 + bz
        y1 = y0 + by
        x1 = x0 + bx
        if z0 < 0 or y0 < 0 or x0 < 0 or z1 > ac.shape[0] or y1 > ac.shape[1] or x1 > ac.shape[2]:
            raise ValueError("boxsize exceeds autocorrelation volume dimensions")
        ac = ac[z0:z1, y0:y1, x0:x1]

    if normalize:
        ac = ac / (ac.max() + 1e-8)

    return ac


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
    shift_reg_subpixel=True,
    shift_reg_subpixel_method="map",
    xpad=None,
    ypad=None,
    fuse_shift=None,
    new_xs=None,
    old_xs=None,
    plane_shifts=None,
    process_mov=False,
    cavity_size=15,
    apply_z_shift=True,
    log_cb=default_log,
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

    mov_shifted = None
    total_batches = int(np.ceil(nt / batch_size))
    for b in range(total_batches):
        mempool.free_all_blocks()
        t1 = b * batch_size  # starting time point of batch
        t2 = int(np.min((nt, (b + 1) * batch_size)))  # end time point of batch

        mov_gpu_full = None
        if process_mov:
            mov_gpu = cp.asarray(mov_cpu[:, t1:t2, :, :])
            mov_gpu, mov_cpu_processed_tmp, mov_gpu_full = process_mov_gpu(
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
        if shift_reg and mov_gpu_full is not None:
            mov_gpu = mov_gpu.copy()

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
            mov_gpu_for_shift = mov_gpu_full if mov_gpu_full is not None else mov_gpu
            shift_start = time.perf_counter()
            # Measurement (sub_pixel_shifts / int_shift) is preserved; we
            # only optionally zero z at the apply step so 2D-effective
            # registration can use the 3D code path without losing the
            # 3D shift estimates from the saved offsets.
            if shift_reg_subpixel:
                shifts_to_apply = sub_pixel_shifts[t1:t2]
                if not apply_z_shift:
                    shifts_to_apply = shifts_to_apply.copy()
                    shifts_to_apply[:, 0] = 0
                if shift_reg_subpixel_method == "fft":
                    mov_gpu_for_shift = shift_gpu_subpixel_fft(
                        mov_gpu_for_shift, shifts_to_apply
                    )
                else:
                    mov_gpu_for_shift = shift_gpu_subpixel_map(
                        mov_gpu_for_shift, shifts_to_apply
                    )
            else:
                shifts_to_apply = int_shift[t1:t2]
                if not apply_z_shift:
                    shifts_to_apply = shifts_to_apply.copy()
                    shifts_to_apply[:, 0] = 0
                mov_gpu_for_shift = shift_gpu(mov_gpu_for_shift, shifts_to_apply)
            shift_elapsed = time.perf_counter() - shift_start
            # log_cb(
            #     f"Shift op ({'subpixel ' + shift_reg_subpixel_method if shift_reg_subpixel else 'integer'}) "
            #     f"batch {b} took {shift_elapsed:.4f}s"
            # )
            if mov_shifted is None:
                mov_shifted = np.zeros(
                    (
                        mov_gpu_for_shift.shape[0],
                        nt,
                        mov_gpu_for_shift.shape[2],
                        mov_gpu_for_shift.shape[3],
                    ),
                    dtype=mov_gpu_for_shift.dtype,
                )
            mov_shifted[:, t1:t2, :, :] = mov_gpu_for_shift.get()

        del mov_gpu
        if mov_gpu_full is not None:
            del mov_gpu_full
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
    shift_reg_subpixel=True,
    shift_reg_subpixel_method="map",
    log_cb=default_log,
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
            shift_start = time.perf_counter()
            if shift_reg_subpixel:
                if shift_reg_subpixel_method == "fft":
                    mov_gpu = shift_gpu_subpixel_fft(mov_gpu, sub_pixel_shifts[t1:t2])
                else:
                    mov_gpu = shift_gpu_subpixel_map(mov_gpu, sub_pixel_shifts[t1:t2])
            else:
                mov_gpu = shift_gpu(mov_gpu, int_shift[t1:t2])
            shift_elapsed = time.perf_counter() - shift_start
            # log_cb(
            #     f"Shift op ({'subpixel ' + shift_reg_subpixel_method if shift_reg_subpixel else 'integer'}) "
            #     f"batch {b} took {shift_elapsed:.4f}s"
            # )
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


def shift_gpu_subpixel_fft(mov_gpu, shifts_zyx):
    """
    Apply subpixel 3D shifts using the Fourier shift theorem.

    Parameters
    ----------
    mov_gpu : cp.ndarray (nz, nt, ny, nx)
    shifts_zyx : array-like (nt, 3)
        Shifts in (z, y, x) order per frame.
    """
    if not HAS_CUPY:
        raise ImportError(
            "GPU subpixel shifting requires cupy. Please install cupy to use this function."
        )

    nz, ntb, ny, nx = mov_gpu.shape
    shifts_zyx = cp.asarray(shifts_zyx, dtype=cp.float32)

    kz = cp.fft.fftfreq(nz).astype(cp.float32)[:, None, None]
    ky = cp.fft.fftfreq(ny).astype(cp.float32)[None, :, None]
    kx = cp.fft.fftfreq(nx).astype(cp.float32)[None, None, :]

    out = cp.empty_like(mov_gpu)
    for t in range(ntb):
        dz, dy, dx = shifts_zyx[t]
        phase = cp.exp(-2j * cp.pi * (dz * kz + dy * ky + dx * kx))
        fft_vol = cufft.fftn(mov_gpu[:, t, :, :], axes=(0, 1, 2))
        shifted = cufft.ifftn(fft_vol * phase, axes=(0, 1, 2)).real
        out[:, t, :, :] = shifted.astype(mov_gpu.dtype, copy=False)

    return out


def shift_gpu_subpixel_map(mov_gpu, shifts_zyx, order=1, mode="constant", cval=0.0):
    """
    Apply subpixel 3D shifts using map_coordinates (trilinear by default).

    Parameters
    ----------
    mov_gpu : cp.ndarray (nz, nt, ny, nx)
    shifts_zyx : array-like (nt, 3)
        Shifts in (z, y, x) order per frame.
    order : int
        Interpolation order (1 = linear).
    mode : str
        Boundary mode for map_coordinates.
    cval : float
        Constant value for padding when mode="constant".
    """
    if not HAS_CUPY:
        raise ImportError(
            "GPU subpixel shifting requires cupy. Please install cupy to use this function."
        )

    nz, ntb, ny, nx = mov_gpu.shape
    shifts_zyx = cp.asarray(shifts_zyx, dtype=cp.float32)

    zz, yy, xx = cp.meshgrid(
        cp.arange(nz, dtype=cp.float32),
        cp.arange(ny, dtype=cp.float32),
        cp.arange(nx, dtype=cp.float32),
        indexing="ij",
    )

    out = cp.empty_like(mov_gpu)
    for t in range(ntb):
        dz, dy, dx = shifts_zyx[t]
        coords = cp.stack([zz + dz, yy + dy, xx + dx])
        out[:, t, :, :] = cuimage.map_coordinates(
            mov_gpu[:, t, :, :],
            coords,
            order=order,
            mode=mode,
            cval=cval,
        ).astype(mov_gpu.dtype, copy=False)

    return out


def clip_mov_gpu(mov, rmin, rmax):
    nz, __, __, __ = mov.shape
    for z in range(nz):
        if rmin[z] is not None and rmax[z] is not None:
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
    mov_gpu_full = mov_gpu
    # crop the movie so only full z-planes count
    if xpad > 0:
        mov_gpu = mov_gpu[:, :, :, xpad:-xpad]
    if ypad > 0:
        mov_gpu = mov_gpu[:, :, ypad:-ypad, :]
    return mov_gpu, mov_cpu_processed_tmp, mov_gpu_full


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


def block_mov_3d(mov_gpu, zblocks, yblocks, xblocks):
    """
    Split a 3D movie into 3D blocks.

    Parameters
    ----------
    mov_gpu : ndarray (nt, nz, ny, nx)
        Movie on GPU, time-first
    zblocks, yblocks, xblocks : list
        Block boundary arrays

    Returns
    -------
    mov_blocks : ndarray (nt, nb, bz, by, bx)
    """
    nt, __, __, __ = mov_gpu.shape
    nb = len(zblocks)
    bz = zblocks[0][1] - zblocks[0][0]
    by = yblocks[0][1] - yblocks[0][0]
    bx = xblocks[0][1] - xblocks[0][0]
    mov_blocks = cp.zeros((nt, nb, bz, by, bx), dtype=mov_gpu.dtype)
    for bidx in range(nb):
        bz0, bz1 = zblocks[bidx]
        by0, by1 = yblocks[bidx]
        bx0, bx1 = xblocks[bidx]
        mov_blocks[:, bidx] = mov_gpu[:, bz0:bz1, by0:by1, bx0:bx1]
    return mov_blocks


def reg_3d_gpu_blocks(mov_blocks, refs_nr_f):
    """
    3D phase correlation for block-wise registration.

    Parameters
    ----------
    mov_blocks : ndarray (nt, nb, bz, by, bx)
    refs_nr_f : ndarray (nb, bz, by, bx)

    Returns
    -------
    phase_corr : ndarray (nt, nb, bz, by, bx)
    """
    refs_nr_f = cp.asarray(refs_nr_f)
    fft_3d_mov = cufft.fftn(mov_blocks, axes=(2, 3, 4))
    fft_3d_mov = fft_3d_mov / (1e-5 + cp.abs(fft_3d_mov))
    fft_3d_mov = fft_3d_mov * refs_nr_f[cp.newaxis, :, :, :, :]
    phase_corr = cp.abs(cufft.ifftn(fft_3d_mov, axes=(2, 3, 4)))
    return phase_corr


def unwrap_fft_3d(mov_float, pc_size, out=None):
    """
    Rearranges the 3D phase correlation so the zero-shift peak is centered.

    The FFT-based phase correlation is periodic; the peak corresponding to a
    negative shift appears at the high end of each axis. This function reorders
    the 8 octants of the correlation volume so that shifts in [-pc_size, +pc_size]
    are centered in the output.

    Parameters
    ----------
    mov_float : ndarray (nt, nb, nz, ny, nx)
        Phase correlation volume
    pc_size : array-like (3,)
        Max shift per axis (z, y, x)
    out : ndarray, optional
        Output array of shape (nt, nb, 2*pc_size+1)

    Returns
    -------
    out : ndarray
        Centered, cropped phase correlation
    """
    nt, nb, nz, ny, nx = mov_float.shape
    pc_size = cp.asarray(pc_size)
    pz = int(pc_size[0])
    py = int(pc_size[1])
    px = int(pc_size[2])
    max_pc_size = n.array([pz, py, px]) * 2 + 1
    if out is None:
        out = cp.zeros(
            (nt, nb, int(max_pc_size[0]), int(max_pc_size[1]), int(max_pc_size[2])),
            dtype=mov_float.dtype,
        )
    # print("Attemptimg to unwrap fft 3d")
    # print("Initial shape:", mov_float.shape)
    # print("Output shape:", out.shape)

    # z+ y+ x+
    out[:, :, pz:, py:, px:] = mov_float[:, :, : pz + 1, : py + 1, : px + 1]
    # z+ y+ x-
    out[:, :, pz:, py:, :px] = mov_float[
        :, :, : pz + 1, : py + 1, nx - px :
    ]
    # z+ y- x+
    out[:, :, pz:, :py, px:] = mov_float[
        :, :, : pz + 1, ny - py :, : px + 1
    ]
    # z+ y- x-
    out[:, :, pz:, :py, :px] = mov_float[
        :, :, : pz + 1, ny - py :, nx - px :
    ]

    # z- y+ x+
    out[:, :, :pz, py:, px:] = mov_float[
        :, :, nz - pz :, : py + 1, : px + 1
    ]
    # z- y+ x-
    out[:, :, :pz, py:, :px] = mov_float[
        :, :, nz - pz :, : py + 1, nx - px :
    ]
    # z- y- x+
    out[:, :, :pz, :py, px:] = mov_float[
        :, :, nz - pz :, ny - py :, : px + 1
    ]
    # z- y- x-
    out[:, :, :pz, :py, :px] = mov_float[
        :, :, nz - pz :, ny - py :, nx - px :
    ]

    return out


def center_phasecorrs(arr, normalize_max=True):
    """
    Center phase-correlation peaks over the last two dimensions.

    Parameters
    ----------
    arr : ndarray
        Input array of arbitrary shape. The last two dimensions are treated
        as the 2D phase-correlation map.

    normalize_max : bool
        If True, divide each map by its maximum so the centered peak is 1.

    Returns
    -------
    ndarray
        Array of same shape as input, with each [*, *] map rolled so its
        maximum is at the center of the last two dimensions.
    """
    if arr.ndim < 2:
        raise ValueError("arr must have at least 2 dimensions")

    xp = cp if (HAS_CUPY and isinstance(arr, cp.ndarray)) else n
    ny, nx = arr.shape[-2], arr.shape[-1]
    cy, cx = ny // 2, nx // 2

    out = arr.copy()
    flat = out.reshape(-1, ny, nx)
    for idx in range(flat.shape[0]):
        argmax = xp.argmax(flat[idx])
        y, x = xp.unravel_index(argmax, (ny, nx))
        shift_y = int(cy - y)
        shift_x = int(cx - x)
        rolled = xp.roll(flat[idx], shift=(shift_y, shift_x), axis=(-2, -1))
        if normalize_max:
            max_val = rolled.max()
            if max_val != 0:
                rolled = rolled / max_val
        flat[idx] = rolled

    return out.reshape(arr.shape)

def compute_fwhm_2d(arr):
    """
    Compute the full-width at half-maximum (FWHM) of 2D peaks in an array.

    Parameters
    ----------
    arr : ndarray (nt, ny, nx)
        Input array containing 2D peaks along the last two dimensions.

    Returns
    -------
    fwhms : ndarray (nt, 2)
        FWHM values along y and x for each 2D peak.
    """
    nt, ny, nx = arr.shape
    fwhms = n.zeros((nt, 2), dtype=n.float32)

    for t in range(nt):
        peak = arr[t]
        half_max = peak.max() / 2.0

        # Y-axis FWHM
        y_indices = n.where(peak >= half_max)[0]
        if y_indices.size > 0:
            fwhm_y = y_indices[-1] - y_indices[0] + 1
        else:
            fwhm_y = 0

        # X-axis FWHM
        x_indices = n.where(peak >= half_max)[1]
        if x_indices.size > 0:
            fwhm_x = x_indices[-1] - x_indices[0] + 1
        else:
            fwhm_x = 0

        fwhms[t, 0] = fwhm_y
        fwhms[t, 1] = fwhm_x

    return fwhms

def get_phasecorr_snr_2d(phasecorrs, npad_xy):
    # a SIMPLE version of get_snr_3d which works for an array of 2D phasecorrs without cupy
    nt, ny, nx = phasecorrs.shape
    snrs = n.zeros(nt, dtype=n.float32)

    for t in range(nt):
        peak = phasecorrs[t]
        half_max = peak.max() / 2.0
        
        # Create padded region mask
        pad_mask = n.zeros_like(peak, dtype=bool)
        if npad_xy > 0:
            pad_mask[:npad_xy, :] = True
            pad_mask[-npad_xy:, :] = True
            pad_mask[:, :npad_xy] = True
            pad_mask[:, -npad_xy:] = True
        
        # Get max in non-padded region
        peak_nopad = peak.copy()
        peak_nopad[pad_mask] = 0
        max_nopad = peak_nopad.max()
        
        # Zero out region around max in padded region
        max_idx = n.unravel_index(n.argmax(peak_nopad), peak.shape)
        y_min = max(0, max_idx[0] - npad_xy)
        y_max = min(ny, max_idx[0] + npad_xy + 1)
        x_min = max(0, max_idx[1] - npad_xy)
        x_max = min(nx, max_idx[1] + npad_xy + 1)
        peak_nopad[y_min:y_max, x_min:x_max] = 0
        
        # Get max in background (outside padded region and around peak)
        max_background = peak_nopad.max()
        
        snrs[t] = max_nopad / n.maximum(1e-10, max_background)

    return snrs


def compute_snr_and_smooth_3d(
    phase_corr, smooth_mat, n_smooth_iters=1, snr_thresh=1.2, npad=3, log_cb=default_log
):
    pc = phase_corr.copy()
    pc_smooth = pc.copy()
    snr0 = get_snr_3d(pc, npad)
    for i in range(n_smooth_iters):
        snrs = get_snr_3d(pc, npad)
        idx_to_smooth = snrs < snr_thresh
        n_low_snr = idx_to_smooth.sum()
        # log_cb("Iter %d: %d/%d blocks below SNR thresh" % (i, n_low_snr, snrs.size), 4)
        if n_low_snr < 1:
            break
        pc_smooth = cp.moveaxis(cp.tensordot(smooth_mat, pc_smooth, ((1,), (1,))), 0, 1)
        pc[idx_to_smooth] = pc_smooth[idx_to_smooth]
    # snrs = get_snr_3d(pc, npad)
    return pc, snr0


def get_snr_3d(phase_corr, npad=3, kernel=None, n_thread_per_block=512, slow=False):
    phase_corr = phase_corr.copy()
    nt, nb, nccz, nccy, nccx = phase_corr.shape
    nball = nt * nb

    if n.isscalar(npad):
        npad = (npad, npad, npad)
    elif len(npad) != 3:
        raise ValueError("npad must be a scalar or a 3-tuple")

    if kernel is None:
        kernel = get_kernel_zero_around_max_3d()

    # print(phase_corr.shape)
    # print(npad)

    # Handle slicing when npad is 0 (0:-0 would be empty, need full slice instead)
    z_slice = slice(npad[0], -npad[0] if npad[0] > 0 else None)
    y_slice = slice(npad[1], -npad[1] if npad[1] > 0 else None)
    x_slice = slice(npad[2], -npad[2] if npad[2] > 0 else None)
    
    # print(z_slice, y_slice, x_slice)
    max_nopad = phase_corr[:, :, z_slice, y_slice, x_slice].max(axis=(-1, -2, -3))

    pc_flat = phase_corr.reshape(nball, nccz * nccy * nccx)
    # print(pc_flat.shape)
    argmaxs = cp.argmax(pc_flat, axis=-1)
    # print(argmaxs.shape)
    argmax_zs, argmax_ys, argmax_xs = cp.unravel_index(argmaxs, (nccz, nccy, nccx))

    xmin = argmax_xs - npad[2]
    xmin[xmin < 0] = 0
    xmax = argmax_xs + npad[2] + 1
    xmax[xmax > nccx] = nccx
    ymin = argmax_ys - npad[1]  
    ymin[ymin < 0] = 0
    ymax = argmax_ys + npad[1] + 1
    ymax[ymax > nccy] = nccy
    zmin = argmax_zs - npad[0]
    zmin[zmin < 0] = 0
    zmax = argmax_zs + npad[0] + 1  # 
    zmax[zmax > nccz] = nccz

    if slow:
        phase_corr = phase_corr.reshape(nball, nccz, nccy, nccx)
        for tid in range(nball):
            phase_corr[
                tid,
                zmin[tid] : zmax[tid],
                ymin[tid] : ymax[tid],
                xmin[tid] : xmax[tid],
            ] = 0
        phase_corr = phase_corr.reshape(nt, nb, nccz, nccy, nccx)
    else:
        phase_corr = phase_corr.reshape(nball, nccz, nccy, nccx)
        n_blocks = int(n.ceil(nball / n_thread_per_block))
        kernel(
            (n_blocks,),
            (n_thread_per_block,),
            (
                zmin,
                zmax,
                ymin,
                ymax,
                xmin,
                xmax,
                phase_corr,
                cp.uint32(nccz),
                cp.uint32(nccy),
                cp.uint32(nccx),
                cp.uint32(nball),
            ),
        )
        phase_corr = phase_corr.reshape(nt, nb, nccz, nccy, nccx)

    max_zerod = phase_corr.max(axis=(-1, -2, -3))
    snrs = max_nopad / cp.maximum(1e-10, max_zerod)
    return snrs


def crop_maxs_3d(pc, npad, kernel=None, n_thread_per_block=512):
    nt, nb, nccz_pad, nccy_pad, nccx_pad = pc.shape
    nball = nb * nt
    if n.isscalar(npad):
        npad = (npad, npad, npad)
    elif len(npad) != 3:
        raise ValueError("npad must be a scalar or a 3-tuple")

    nccz_nopad = nccz_pad - npad[0] * 2
    nccy_nopad = nccy_pad - npad[1] * 2
    nccx_nopad = nccx_pad - npad[2] * 2

    pc_nopad = cp.zeros(
        (nt, nb, nccz_nopad, nccy_nopad, nccx_nopad), dtype=cp.float32
    )
    # Handle slicing when npad is 0 (0:-0 would be empty, need full slice instead)
    z_slice = slice(npad[0], -npad[0] if npad[0] > 0 else None)
    y_slice = slice(npad[1], -npad[1] if npad[1] > 0 else None)
    x_slice = slice(npad[2], -npad[2] if npad[2] > 0 else None)
    pc_nopad[:] = pc[:, :, z_slice, y_slice, x_slice]

    argmaxs = cp.argmax(pc_nopad.reshape(nt, nb, -1), axis=-1)
    zmaxs, ymaxs, xmaxs = cp.unravel_index(
        argmaxs, (nccz_nopad, nccy_nopad, nccx_nopad)
    )

    xmin = xmaxs - npad[2]
    xmin[xmin < 0] = 0
    xmax = xmaxs + npad[2] + 1
    xmax[xmax > nccx_nopad] = nccx_nopad
    ymin = ymaxs - npad[1]
    ymin[ymin < 0] = 0
    ymax = ymaxs + npad[1] + 1
    ymax[ymax > nccy_nopad] = nccy_nopad
    zmin = zmaxs - npad[0]
    zmin[zmin < 0] = 0
    zmax = zmaxs + npad[0] + 1
    zmax[zmax > nccz_nopad] = nccz_nopad

    if kernel is None:
        kernel = get_kernel_crop_around_max_3d()

    npadmat_z = npad[0] * 2 + 1
    npadmat_y = npad[1] * 2 + 1
    npadmat_x = npad[2] * 2 + 1

    pc_mat = cp.zeros((nt, nb, npadmat_z, npadmat_y, npadmat_x), cp.float32)

    use_kernel = npad[0] == npad[1] == npad[2]
    if use_kernel:
        npadmat = npadmat_z
        ncc_nopad = nccz_nopad
        n_blocks = int(n.ceil(nball / n_thread_per_block))
        kernel(
            (n_blocks,),
            (n_thread_per_block,),
            (
                zmin.ravel(),
                zmax.ravel(),
                ymin.ravel(),
                ymax.ravel(),
                xmin.ravel(),
                xmax.ravel(),
                pc_nopad.reshape(nball, ncc_nopad, ncc_nopad, ncc_nopad),
                pc_mat.reshape(nball, npadmat, npadmat, npadmat),
                cp.uint32(npadmat),
                cp.uint32(ncc_nopad),
                cp.uint32(nball),
            ),
        )
    else:
        zmin_h = cp.asnumpy(zmin.ravel())
        zmax_h = cp.asnumpy(zmax.ravel())
        ymin_h = cp.asnumpy(ymin.ravel())
        ymax_h = cp.asnumpy(ymax.ravel())
        xmin_h = cp.asnumpy(xmin.ravel())
        xmax_h = cp.asnumpy(xmax.ravel())
        pc_nopad_h = pc_nopad.reshape(nball, nccz_nopad, nccy_nopad, nccx_nopad)
        pc_mat_h = pc_mat.reshape(nball, npadmat_z, npadmat_y, npadmat_x)
        for tid in range(nball):
            pc_mat_h[tid, : zmax_h[tid] - zmin_h[tid], : ymax_h[tid] - ymin_h[tid], : xmax_h[tid] - xmin_h[tid]] = (
                pc_nopad_h[tid, zmin_h[tid] : zmax_h[tid], ymin_h[tid] : ymax_h[tid], xmin_h[tid] : xmax_h[tid]]
            )

    return pc_mat, zmaxs, ymaxs, xmaxs


def get_kernel_crop_around_max_3d():
    kernel_crop_around_max = cp.RawKernel(
        r"""
    extern "C" __global__
    void crop_around_max_3d(long long* zmin, long long* zmax, long long* ymin, long long* ymax,
                            long long* xmin, long long* xmax, float* in, float* out,
                            unsigned int npadmat, unsigned int ncc, unsigned int max){
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int xx; int yy; int zz;
        int i; int j; int k;
        if (tid < max){
            i = 0;
            for (zz = zmin[tid]; zz < zmax[tid]; zz++){
                j = 0;
                for (yy = ymin[tid]; yy < ymax[tid]; yy++){
                    k = 0;
                    for (xx = xmin[tid]; xx < xmax[tid]; xx++){
                        out[(tid * npadmat * npadmat * npadmat) + (i * npadmat * npadmat) + (j * npadmat) + k] =
                            in[(tid * ncc * ncc * ncc) + (zz * ncc * ncc) + (yy * ncc) + xx];
                        k++;
                    }
                    j++;
                }
                i++;
            }
        }
    }
    """,
        "crop_around_max_3d",
    )
    return kernel_crop_around_max


def get_kernel_zero_around_max_3d():
    kernel_zero_around_max = cp.RawKernel(
        r"""
    extern "C" __global__
    void zero_around_max_3d(long long* zmin, long long* zmax, long long* ymin, long long* ymax,
                            long long* xmin, long long* xmax, float* out,
                            unsigned int nccz, unsigned int nccy, unsigned int nccx,
                            unsigned int max){
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int xx; int yy; int zz;
        if (tid < max){
            for (zz = zmin[tid]; zz < zmax[tid]; zz++){
                for (yy = ymin[tid]; yy < ymax[tid]; yy++){
                    for (xx = xmin[tid]; xx < xmax[tid]; xx++){
                        out[(tid * nccz * nccy * nccx) + (zz * nccy * nccx) + (yy * nccx) + xx] = 0;
                    }
                }
            }
        }
    }
    """,
        "zero_around_max_3d",
    )
    return kernel_zero_around_max


def kernelD1d(xs: n.ndarray, ys: n.ndarray, sigL: float = 0.85) -> n.ndarray:
    xs0 = xs.reshape(-1, 1)
    ys0 = ys.reshape(1, -1)
    dxs = xs0 - ys0
    K = n.exp(-(dxs**2) / (2 * sigL**2))
    return K


@lru_cache(maxsize=5)
def mat_upsample_1d(lpad: int, subpixel: int = 10):
    lar = n.arange(-lpad, lpad + 1)
    larUP = n.arange(-lpad, lpad + 0.001, 1.0 / subpixel)
    nup = larUP.shape[0]
    Kmat = n.linalg.inv(kernelD1d(lar, lar)) @ kernelD1d(lar, larUP)
    return Kmat, nup


def get_subpixel_shifts_3d(
    pc, max_shift, npad=3, subpixel=5, n_thread_per_block=512
):
    # if npad is an int, make it a 3-tuple
    if n.isscalar(npad):
        npad = (npad, npad, npad)

    if n.isscalar(subpixel):
        subpixel = (subpixel, subpixel, subpixel)
    nt, nb = pc.shape[:2]
    Kz, nupz = mat_upsample_1d(lpad=npad[0], subpixel=subpixel[0])
    Ky, nupy = mat_upsample_1d(lpad=npad[1], subpixel=subpixel[1])
    Kx, nupx = mat_upsample_1d(lpad=npad[2], subpixel=subpixel[2])
    Kz = cp.asarray(Kz, cp.float32)
    Ky = cp.asarray(Ky, cp.float32)
    Kx = cp.asarray(Kx, cp.float32)

    pc_mat, zmaxs, ymaxs, xmaxs = crop_maxs_3d(
        pc, npad, n_thread_per_block=n_thread_per_block
    )

    zmaxs = zmaxs - max_shift[0]
    ymaxs = ymaxs - max_shift[1]
    xmaxs = xmaxs - max_shift[2]

    # separable upsampling: z -> y -> x
    pc_up = cp.tensordot(pc_mat, Kz, axes=([2], [0]))
    pc_up = cp.moveaxis(pc_up, -1, 2)
    pc_up = cp.tensordot(pc_up, Ky, axes=([3], [0]))
    pc_up = cp.moveaxis(pc_up, -1, 3)
    pc_up = cp.tensordot(pc_up, Kx, axes=([4], [0]))

    argmaxs = cp.argmax(pc_up.reshape(nt, nb, -1), axis=-1)
    zmaxs_sub, ymaxs_sub, xmaxs_sub = cp.unravel_index(argmaxs, (nupz, nupy, nupx))

    midz = nupz // 2
    midy = nupy // 2
    midx = nupx // 2

    zmaxs = zmaxs.astype(cp.float32) + (zmaxs_sub.astype(cp.float32) - midz) / subpixel[0]
    ymaxs = ymaxs.astype(cp.float32) + (ymaxs_sub.astype(cp.float32) - midy) / subpixel[1]
    xmaxs = xmaxs.astype(cp.float32) + (xmaxs_sub.astype(cp.float32) - midx) / subpixel[2]

    return zmaxs, ymaxs, xmaxs


def nonrigid_3d_gpu(
    mov_cpu,
    mult_mask,
    add_mask,
    refs_nr_f,
    zblocks,
    yblocks,
    xblocks,
    snr_thresh,
    smooth_mat,
    max_shift,
    rmins=None,
    rmaxs=None,
    npad=3,
    n_smooth_iters=1,
    subpixel=5,
    n_gpu_threads_per_block=512,
    batch_size=20,
    log_cb=default_log,
    save_phasecorrs=False,
):
    """
    Nonrigid 3D registration on the GPU, returning a 3D shift per block.

    Parameters
    ----------
    mov_cpu : ndarray (nz, nt, ny, nx)
        Movie (z, time, y, x)
    mult_mask, add_mask : ndarray (nb, bz, by, bx)
        Block-wise masks
    refs_nr_f : ndarray (nb, bz, by, bx)
        Block-wise FFT references
    zblocks, yblocks, xblocks : list
        Block boundaries
    snr_thresh : float
        SNR threshold for smoothing
    smooth_mat : ndarray (nb, nb)
        Nonrigid smoothing matrix
    max_shift : array-like (3,)
        Max shift per axis (z, y, x)

    Returns
    -------
    zshifts, yshifts, xshifts : ndarray
        (nt, nzb, nyb, nxb) subpixel shifts per block
    snrs : ndarray
        (nt, nzb, nyb, nxb) SNR per block
    """
    if not HAS_CUPY:
        raise ImportError(
            "GPU registration requires cupy. Please install cupy to use this function."
            " See https://docs.cupy.dev/en/stable/install.html for installation instructions."
        )

    mempool = cp.get_default_memory_pool()
    nz, nt, __, __ = mov_cpu.shape
    nb = len(zblocks)

    if n.isscalar(max_shift):
        max_shift = n.array([max_shift, max_shift, max_shift])
    else:
        max_shift = n.asarray(max_shift)
    if n.isscalar(npad):
        npad = n.array([npad, npad, npad])
    else:
        npad = n.asarray(npad)
        if npad.size != 3:
            raise ValueError("npad must be a scalar or a 3-tuple")

    # unwrap_fft_3d requires (max_shift + npad) + 1 <= block_size per axis.
    # If not, the centered-window slicing silently runs off the end of the
    # block and broadcast-errors at assignment. Clamp here with a loud
    # warning; shrink npad (parabolic-fit margin) before max_shift.
    block_size = n.array([
        zblocks[0][1] - zblocks[0][0],
        yblocks[0][1] - yblocks[0][0],
        xblocks[0][1] - xblocks[0][0],
    ])
    nr = max_shift + npad
    if (nr + 1 > block_size).any():
        max_nr = block_size - 1
        nr_clamped = n.minimum(nr, max_nr)
        npad_new = n.minimum(npad, n.maximum(nr_clamped - max_shift, 0))
        max_shift_new = n.minimum(max_shift, nr_clamped - npad_new)
        bad = n.where(nr + 1 > block_size)[0].tolist()
        log_cb(
            "WARNING: nonrigid phase-corr window exceeds block_size_3d on "
            "axis(es) %s. Clamping max_shift_nr %s -> %s, nr_npad %s -> %s. "
            "Increase block_size_3d to recover dynamic range."
            % (bad, max_shift.tolist(), max_shift_new.tolist(),
               npad.tolist(), npad_new.tolist()),
            0,
        )
        max_shift = max_shift_new
        npad = npad_new

    nr = max_shift + npad
    ncc = nr * 2 + 1

    log_cb("Nonrigid 3D registration on GPU:", 3)
    log_cb(" Movie size: %s" % (str(mov_cpu.shape)), 4)
    log_cb(" Number of blocks: %d" % (nb), 4)
    log_cb(" Max shifts (z,y,x): %s" % (str(max_shift)), 4)
    log_cb(" Npad (z,y,x): %s" % (str(npad)), 4)
    log_cb(" Phase corr size (z,y,x): %s" % (str(ncc)), 4)

    zstarts = n.array([zb[0] for zb in zblocks])
    ystarts = n.array([yb[0] for yb in yblocks])
    xstarts = n.array([xb[0] for xb in xblocks])
    nzb = n.unique(zstarts).shape[0]
    nyb = n.unique(ystarts).shape[0]
    nxb = n.unique(xstarts).shape[0]
    # print(nzb, nyb, nxb)
    # print(nb)
    if nb != nzb * nyb * nxb:
        raise ValueError("Block list length does not match block grid dimensions")

    mult_mask = cp.asarray(mult_mask, cp.float32)
    add_mask = cp.asarray(add_mask, cp.float32)
    refs_nr_f = cp.asarray(refs_nr_f, cp.complex64)
    smooth_mat = cp.asarray(smooth_mat, cp.float32)

    zshifts = cp.zeros((nt, nb), dtype=cp.float32)
    yshifts = cp.zeros((nt, nb), dtype=cp.float32)
    xshifts = cp.zeros((nt, nb), dtype=cp.float32)
    snrs = cp.zeros((nt, nb), dtype=cp.float32)
    phase_corrs = None
    if save_phasecorrs:
        phase_corrs = cp.zeros(
            (nt,2, nb, int(ncc[0]), int(ncc[1]), int(ncc[2])), dtype=cp.float32
        )

    total_batches = int(n.ceil(nt / batch_size))
    for b in range(total_batches):
        mempool.free_all_blocks()
        t1 = b * batch_size
        t2 = int(n.min((nt, (b + 1) * batch_size)))

        mov_gpu = cp.asarray(mov_cpu[:, t1:t2, :, :])
        if rmins is not None and rmaxs is not None:
            # print("CLIPPIGN")
            # print(rmins, rmaxs)
            mov_gpu = clip_mov_gpu(mov_gpu, rmins, rmaxs)

        # switch to time-first for blocking
        mov_gpu = mov_gpu.swapaxes(0, 1)

        mov_blocks = block_mov_3d(mov_gpu, zblocks, yblocks, xblocks)
        mov_blocks *= mult_mask[cp.newaxis, :, :, :, :]
        mov_blocks += add_mask[cp.newaxis, :, :, :, :]

        phase_corr = reg_3d_gpu_blocks(mov_blocks, refs_nr_f)
        # print(phase_corr.shape)
        phase_corr = unwrap_fft_3d(phase_corr, nr)        
        if save_phasecorrs:
            phase_corrs[t1:t2, 0] = phase_corr.copy()

        # TODO: consider iterative smoothing (multiple passes) after initial validation
        pc, snr_batch = compute_snr_and_smooth_3d(
            phase_corr, smooth_mat, n_smooth_iters, snr_thresh, log_cb=log_cb, npad=npad
        )

        if save_phasecorrs:
            phase_corrs[t1:t2, 1] = pc

        zsub, ysub, xsub = get_subpixel_shifts_3d(
            pc, max_shift, npad, subpixel, n_gpu_threads_per_block
        )

        zshifts[t1:t2] = zsub
        yshifts[t1:t2] = ysub
        xshifts[t1:t2] = xsub
        snrs[t1:t2] = snr_batch

        del mov_gpu
        del mov_blocks
        del phase_corr
        mempool.free_all_blocks()

    zshifts = zshifts.reshape(nt, nzb, nyb, nxb)
    yshifts = yshifts.reshape(nt, nzb, nyb, nxb)
    xshifts = xshifts.reshape(nt, nzb, nyb, nxb)
    snrs = snrs.reshape(nt, nzb, nyb, nxb)

    if save_phasecorrs:
        return zshifts, yshifts, xshifts, snrs, phase_corrs
    else:
        return zshifts, yshifts, xshifts, snrs, None


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
