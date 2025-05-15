# Function used for creating the reference image
# It will have functions from suite2P which have been slightly modified
import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.fft as cufft
    import scipy.fft
    import cupyx.scipy.ndimage as cuimage
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = np

import time
try:
    from mkl_fft import fft2
except ImportError:
    from scipy.fft import fft2
from numba import vectorize
from numpy import fft


from . import register_gpu as reg
from . import reg_3d as reg_3d  # new 3d registration functions
from .utils import default_log

n = np

# Function which runs createion of reference image+ masks

# There are function at the bottom of this page for controlling the 3D reg version of reference img creation.
def compute_reference_and_masks(
    mov_fuse, reference_params, log_cb=default_log, rmins=None, rmaxs=None, use_GPU=True
):
    """
    Runs the creation of the 3D reference image and creates masks for rigid and non0rigid registration

    Parameters
    ----------
    mov_fuse : ndarray (nz, nt, ny, nx)
        The section of the movie used to create the reference image and masks, which has been fused if needed
    fuse_shift : int
        size of overlapping x planes needed to be fused
    xs : ndarray
        The starting points of the x-patches
    rmins : ndarray (nz), optional
        The minimum pixel values allowed, by default None
    rmaxs : ndarray (nz), optional
        the Maximum pixel values allowed, by default None
    percent_contribute : float, optional
        The percentage of frames used to contribute to the reference iamge, by default 0.9
    niter : int, optional
        The number of iteration used to create the reference image reccomend (4-12), by default 8
    use_GPU : bool, optional
        If tru use the gpu for registration to speed up run-time, by default True

    Returns
    -------
    tvecs : list of arrays
        The plane shifts as a list [correcteded for refernce shift, uncorrrected]
    ref_image : ndarray (nz, ny, nx)
        The padded, and z-alligned reference image
    ref_padded : ndarray (nz, ny, nx)
        The padded NOT alligned reference image
    all_refs_and_masks : nested list
        see get_phasecorr_and_masks function
    pad_sizes : list
        The [x,y] pad sizes
    refernce_params:
        The params used for creating the reference image
    """

    percent_contribute = reference_params["percent_contribute"]
    niter = reference_params["niter"]
    sigma = reference_params["sigma"]
    max_reg_xy = reference_params["max_reg_xy_reference"]

    nz, nt, ny, nx = mov_fuse.shape

    if use_GPU:
        batch_size = reference_params["batch_size"]

    # Run the reference image creation
    if use_GPU:
        ref_image, ymax, xmax, cmax, used_frames = get_reference_img_gpu(
            mov_fuse,
            percent_contribute,
            niter,
            rmins=rmins,
            rmaxs=rmaxs,
            batch_size=batch_size,
            max_reg_xy=max_reg_xy,
            sigma=sigma,
        )
    else:
        ref_image, ymax, xmax, cmax, used_frames = get_reference_img_cpu(
            mov_fuse,
            percent_contribute,
            niter,
            rmins=rmins,
            rmaxs=rmaxs,
            max_reg_xy=max_reg_xy,
            sigma=sigma,
        )

    if reference_params.get("plane_to_plane_alignment", True):
        # allign the reference image
        uncorrected_tvecs = align_planes(ref_image, reference_params)
        # correct bad tvec estimates
        if reference_params.get("fix_shallow_plane_shift_estimates", True):
            shallow_plane_thresh = reference_params.get(
                "fix_shallow_plane_shift_esimate_threshold", 20
            )
            peaks = n.abs(uncorrected_tvecs[:shallow_plane_thresh]).max(axis=0)
            bad_planes = n.logical_or(
                n.abs(uncorrected_tvecs[shallow_plane_thresh:, 0]) > peaks[0],
                n.abs(uncorrected_tvecs[shallow_plane_thresh:, 1]) > peaks[1],
            )
            uncorrected_tvecs[shallow_plane_thresh:][bad_planes, :] = 0
            if bad_planes.sum() > 0:
                log_cb("Fixing %d plane alignment outliers" % bad_planes.sum(), 2)

        # Find the mean shift for the used frames
        xshift = np.zeros(nz)
        yshift = np.zeros(nz)
        for i in range(niter):
            for z in range(nz):
                xshift[z] += xmax[i, z, np.asarray(used_frames[i])[z]].mean(axis=0)
                yshift[z] += ymax[i, z, np.asarray(used_frames[i])[z]].mean(axis=0)

        # allign from raw data
        corrected_tvecs_x = uncorrected_tvecs[:, 1] + xshift - xshift[0]
        corrected_tvecs_y = (
            uncorrected_tvecs[:, 0] - yshift + yshift[0]
        )  # Make the tvecs start at 0!
        corrected_tvecs = np.stack((corrected_tvecs_y, corrected_tvecs_x), axis=1)
        corrected_tvecs = np.round(corrected_tvecs)
        # pad the movie byusing the maximum of either tvecs
        stack_tmp = np.stack((corrected_tvecs, uncorrected_tvecs))
        max_tvecs = np.max(np.abs(stack_tmp), axis=0)
        max_tvecs *= np.sign(
            corrected_tvecs
        )  # may want to change this, the point is to return correct sign after abs

        ref_padded, xpad, ypad = pad_mov3D(ref_image, max_tvecs)
        pad_sizes = [xpad, ypad]

    else:
        uncorrected_tvecs = np.zeros((nz, 2))

    ref_image = apply_plane_shifts3D(ref_padded, uncorrected_tvecs)

    # Option to clip the ref_image per plane for below
    plane_mins = np.zeros(nz)
    plane_maxs = np.zeros(nz)
    clipped_ref_img = np.zeros_like(ref_image)
    for z in range(nz):
        rmin, rmax = n.int16(n.percentile(ref_image[z], 1)), n.int16(
            n.percentile(ref_image[z], 99)
        )
        clipped_ref_img[z] = n.clip(ref_image[z], rmin, rmax)
        plane_mins[z], plane_maxs[z] = rmin, rmax

    # add useful info to reference params
    reference_params["plane_mins"] = plane_mins
    reference_params["plane_maxs"] = plane_maxs
    reference_params["ymax"] = ymax
    reference_params["xmax"] = xmax
    reference_params["cmax"] = cmax
    reference_params["used_frames"] = used_frames

    if reference_params.get("norm_frames", True):
        all_refs_and_masks, reference_params = get_phasecorr_and_masks(
            clipped_ref_img, reference_params
        )
    else:
        all_refs_and_masks, reference_params = get_phasecorr_and_masks(
            ref_image, reference_params
        )

    tvecs = [corrected_tvecs, uncorrected_tvecs]  # return both
    return tvecs, ref_image, ref_padded, all_refs_and_masks, pad_sizes, reference_params


def get_phasecorr_and_masks(ref_image, reference_params):
    """
    Produces all the needed fft'd reference images and masks needed for rigid and non-rigid registration.

    Parameters
    ----------
    ref_image : ndarray (nz, ny, nx)
        A 3D movie, reccomened to be the created reference image

    Returns
    -------
    all_refs_and_masks nested list
        A lsit the first dimeniosn is nz, then for each plane there are six arrays for the masks and fft'd refernces
    """

    nz, Ly, Lx = ref_image.shape
    # Rigid phasscorr and masks
    sigma = reference_params["sigma"]
    block_size = reference_params["block_size"]
    smooth_sigma = reference_params["smooth_sigma"]

    mult_mask_rigid, add_mask_rigid = compute_masks3D(ref_image.squeeze(), sigma)
    refs_fft_rigid = np.zeros_like(ref_image, dtype=complex)
    for z in range(nz):
        refs_fft_rigid[z] = phasecorr_ref(
            ref_image[z, :, :].squeeze(), smooth_sigma=smooth_sigma
        )

    # Non-rgid phascorr refernce and masks
    yblock, xblock, nblocks, block_size, reference_params["NRsm"] = make_blocks(
        Ly=Ly, Lx=Lx, block_size=block_size
    )
    # these params are needed for registration
    reference_params["nblocks"] = nblocks
    reference_params["xblock"] = xblock
    reference_params["yblock"] = yblock

    mult_mask_NR = np.zeros(
        (nblocks[0] * nblocks[1], nz, 1, block_size[0], block_size[1])
    )
    add_mask_NR = np.zeros((nblocks[0] * nblocks[1], nz, 1, block_size[0], block_size[1]))
    refs_fft_NR = np.zeros(
        (nblocks[0] * nblocks[1], nz, 1, block_size[0], block_size[1]), dtype=complex
    )
    for z in range(nz):
        mult_mask_NR[:, z, :, :], add_mask_NR[:, z, :, :], refs_fft_NR[:, z, :, :] = (
            phasecorr_reference(
                refImg0=ref_image[z],
                maskSlope=smooth_sigma * 3,
                smooth_sigma=smooth_sigma,
                yblock=yblock,
                xblock=xblock,
            )
        )

    # created the all_refs_and_maks nested list used for future registration
    all_refs_and_masks = []
    for z in range(nz):
        a_plane_refs_and_masks = []

        # NOTE this is the order of all_refs_and_maks if needed to know
        a_plane_refs_and_masks.append(mult_mask_rigid[z])
        a_plane_refs_and_masks.append(add_mask_rigid[z])
        a_plane_refs_and_masks.append(refs_fft_rigid[z])
        a_plane_refs_and_masks.append(mult_mask_NR[:, z, :, :])
        a_plane_refs_and_masks.append(add_mask_NR[:, z, :, :])
        a_plane_refs_and_masks.append(refs_fft_NR[:, z, :, :])

        all_refs_and_masks.append(a_plane_refs_and_masks)

    return all_refs_and_masks, reference_params


# The 3D version of the reference image
def init_ref_3d(frames):
    """
    Returns a intial reference image, based on the 20 most correlate 3D volumes in frames.
    This fucntion is a 3D version of a Suite 2P function.

    Parameters
    ----------
    frames: Lz x nt z Ly x Lx
        A selection of frames which is used to create the reference image

    Returns
    -------
    refImg: Lz x Ly x Lx
        A inital reference image.
    """

    # want (nt, nz,ny,nx)
    frames = np.swapaxes(frames, 0, 1)
    nimg, nz, ny, nx = frames.shape

    frames = np.reshape(frames, (nimg, -1)).astype(
        "float32"
    )  # flattend all spatial dimensions
    frames -= np.reshape(
        frames.mean(axis=1), (nimg, 1)
    )  # subtract mean intestity at each time
    cc = np.matmul(frames, frames.T)
    ndiag = np.sqrt(np.diag(cc))
    cc /= np.outer(ndiag, ndiag)  # normalised correlation matrix
    CCsort = -np.sort(-cc, axis=1)

    bestCC = np.mean(CCsort[:, 1:20], axis=1)

    imax = np.argmax(
        bestCC
    )  # choose the time frame, which is most correlated with the other frames
    indsort = np.argsort(
        -cc[imax, :]
    )  # gets the correlation if imax with all the other time points
    refImg = np.mean(
        frames[indsort[0:20], :], axis=0
    )  # chose 20 most correlated images to the highly correlated img as the initial ref
    refImg = np.reshape(refImg, (nz, ny, nx))

    return refImg


# function adapted from suite2p made to be 3D
def mean_centered_meshgrid3D(nz, ny, nx):
    """
    Returns a 3D mean-centered meshgrid.
    This function is adapted from suite 2p

    Parameters
    ----------
    nz: int
        length of the z axis
    ny: int
        length of the y axis
    nx: int
        length of the x axis

    Returns
    -------
    zz: ndarray
    yy: ndarray
    xx: ndarray
    """

    x = np.arange(0, nx)
    y = np.arange(0, ny)
    z = np.arange(0, nz)

    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    z = np.abs(z - z.mean())

    zz, yy, xx = np.meshgrid(
        z, y, x, indexing="ij"
    )  # change order here for ideal shape of mask
    return zz, yy, xx


# Suite2p function
def meshgrid_mean_centered(x, y):
    """
    Returns a 2D mean-centered meshgrid.
    This function taken from suite 2p.

    Parameters
    ----------
    x: int
        The height of the meshgrid
    y: int
        The width of the mehgrid

    Returns
    -------
    xx: int array
    yy: int array
    """
    x = np.arange(0, x)
    y = np.arange(0, y)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    return xx, yy


# in to suite 3D
def gaussian_fft(sig, ny, nx):
    """
    Returns a gaussian filter in the Fourier domain std sig and size (ny, nx).
    This function is adapted from suite 2p

    Parameters
    ----------
    sig: float
        standard deviation of the gaussian
    ny: int
        length of the y axis
    nx: int
        length of the x axis

    Returns
    -------
    fhg: ndarray
        gussian filter in the Fourier domain
    """

    # need 2D x/y mesh grid
    xx, yy = meshgrid_mean_centered(nx, ny)

    hgx = n.exp(-n.square(xx / sig) / 2)
    hgy = n.exp(-n.square(yy / sig) / 2)
    hgg = hgy * hgx
    hgg /= hgg.sum()
    fhg = n.real(fft2(n.fft.ifftshift(hgg)))
    return fhg


# currently done 2-D
def phasecorr_ref(RefImg, smooth_sigma=None):
    """
    Returns reference image fft"ed and complex conjugate and multiplied by gaussian filter in the fft domain,
    with standard deviation "smooth_sigma" computes fft"ed reference image for phasecorr.
    This function is adapted from suite 2p

    Parameters
    ----------
    refImg: 2D array (ny, nx)
        reference image
    smooth_sigma: float
        standard deviation of the gaussian

    Returns
    -------
    cfRefImg : 2D array, complex64
    """
    # ny, nx = RefImg.shape,  #padding?
    # get the comple conjugate og the fft'd reference image
    cfRefImg = n.conj(
        fft2(RefImg)
    )  # padding?  , (next_fast_len(ny), next_fast_len(nx))))
    # normalise cfRefImg
    cfRefImg /= 1e-5 + n.absolute(cfRefImg)
    cfRefImg = cfRefImg * gaussian_fft(smooth_sigma, cfRefImg.shape[0], cfRefImg.shape[1])
    return cfRefImg.astype("complex64")


def spatial_taper3D(sig, sigz, nz, ny, nx):
    """
    Returns the multiplication mask need for registration.
    This function is adapted from suite 2p

    Parameters
    ----------
    sig: float
        value of sigma for the x/y direction
    sigz: float
        value of sigma for the z direction (input 0 to skip), should be small
    nz: int
        length of the z axis
    ny: int
        length of the y axis
    nx: int
        length of the x axis

    Returns
    -------
    maks_mul: ndarray
        multiplacation mask needed for registration
    """
    zz, yy, xx = mean_centered_meshgrid3D(nz, ny, nx)

    mY = ((ny - 1) / 2) - 2 * sig
    mX = ((nx - 1) / 2) - 2 * sig
    mZ = ((nz - 1) / 2) - 2 * sigz  # may want to change the Z axis..
    maskY = 1.0 / (1.0 + np.exp((yy - mY) / sig))
    maskX = 1.0 / (1.0 + np.exp((xx - mX) / sig))
    # may not want a z axis taper
    if sigz == 0:
        mask_mul = maskY * maskX
    else:
        maskZ = 1.0 / (1.0 + np.exp((zz - mZ) / sigz))
        mask_mul = maskY * maskX * maskZ

    return mask_mul


def compute_masks3D(refImg, sigma):
    """
    Returns the masks need for registration.

    Parameters
    ----------
    refImg: ndarray (nz, ny, nx)
        The referenace image used for registration
    sigma: tuple [sig, sigz]
        value of sigma for thex/y direction and z direction

    Returns
    -------
    maks_mul: ndarray
        multiplacation mask needed for registration
    mask_offset: ndarray
        offset mask needed for registration
    """
    sig, sigz = sigma
    nz, ny, nx = refImg.shape
    mask_mul = spatial_taper3D(sig, sigz, nz, ny, nx)
    mask_offset = refImg.mean() * (1.0 - mask_mul)
    return mask_mul, mask_offset


def compute_mask_offset(refImg, mask_mul):
    """
    Seperate function as it allows calling this in the reference image loop, as maks_mul only depends
    on the shpae of the array, therefore doesnt need to be recalculated.

    Parameters
    ----------
    refImg: ndarray (nz, ny, nx)
        The referenace image used for registration
    maks_mul: ndarray
        multiplacation mask needed for registration

    Returns
    -------
    mask_offset: ndarray
        offset mask needed for registration
    """
    mask_offset = refImg.mean() * (1.0 - mask_mul)
    return mask_offset


# TODO look at this function
@vectorize(nopython=True, target="parallel", cache=True)
def apply_mask(data, mask_mul, mask_offset):
    """
    Applies the mask to the data

    Parameters
    ----------
    data: ndarray (nt, nz, ny, nx)
        The data used for registration
    maks_mul: ndarray
        multiplacation mask needed for registration
    mask_offset: ndarray
        offset mask needed for registration

    Returns
    -------
    data: ndarray(nt, nz, ny, nx)
        the data with the multiplication and offset masks applied
    """
    return np.complex64(np.float32(data) * mask_mul + mask_offset)


###############################################################################
# New functions needed for creating a reference image


def align_planes(mov3D, reference_params):
    """
    Input a (nz, ny, nx) movie and this function will find the planeshifts between z-axis
    """
    # print("FUNC CALL")
    sigma = reference_params["sigma"]
    smooth_sigma = reference_params["smooth_sigma"]
    max_reg_xy = reference_params["max_reg_xy_reference"]

    mov3D = n.asarray(mov3D, dtype=n.complex64)
    mov3D = np.expand_dims(
        mov3D, axis=1
    )  # make it (nz, 1, ny, nx) so it is in for used by other registration function

    # set up params
    ncc = max_reg_xy * 2 + 1
    nz, nt, ny, nx = mov3D.shape
    # print("GOT SHAPE")
    ymaxs = n.zeros((nz, nt), dtype=n.int16)
    xmaxs = n.zeros((nz, nt), dtype=n.int16)
    cmaxs = n.zeros((nz, nt), dtype=n.float32)
    ncc = max_reg_xy * 2 + 1
    phase_corr = n.zeros((nt, ncc, ncc))

    # print("COMPUTING MASKS")
    mult_mask, add_mask = compute_masks3D(mov3D.squeeze(), sigma)
    # print("DONE")

    # turn the input mov into a reference image for registration
    refs_f = np.zeros_like(mov3D)
    # print("LOOP1")
    for z in range(nz):
        # print(z)
        refs_f[z] = phasecorr_ref(mov3D[z, :, :].squeeze(), smooth_sigma=smooth_sigma)
    # return None

    # find the shifts between two z planes
    for zidx in range(1, nz):
        # print(z)
        mov3D[zidx] = reg.clip_and_mask_mov(
            mov3D[zidx],
            None,
            None,  # can speed thisup with numba/parallelisation?
            mult_mask[zidx],
            add_mask[zidx],
            cp=n,
        )
        mov3D[zidx] = reg.convolve_2d_cpu(
            mov3D[zidx], refs_f[zidx - 1]
        )  # here is zidx and zidx - 1
        reg.unwrap_fft_2d(mov3D[zidx].real, max_reg_xy, out=phase_corr, cp=n)
        ymaxs[zidx], xmaxs[zidx], cmaxs[zidx] = reg.get_max_cc_coord(
            phase_corr, max_reg_xy, cp=n
        )

    tvecY = -np.cumsum(ymaxs)
    tvecX = -np.cumsum(xmaxs)
    return np.stack((tvecY, tvecX), axis=1)


# TODO for these function apply_plane_shiftsXD, this can take a bit of time ~30-60s for a full 4D movie
# There should be a faster way, numba doesnt allow 2D shifts, joblib was not faster over time for 4D array
def apply_plane_shifts3D(mov, tvecs):
    """
    Applies planeshifts over the Z axis to allign a padded movie

    Parameters
    ----------
    mov : ndarray (nz, ny, nx)
        A padded 3D array
    tvecs : ndarray
        y/x pixel shift values for each z-plane

    Returns
    -------
    shifted_mov : ndarray
        Shifted version on the input movie
    """
    nz = mov.shape[0]

    shifted_mov = np.zeros_like(mov)
    for i in range(nz):
        shifted_mov[i] = reg.shift_frame(
            mov[i], -tvecs[i, 0].astype(int), -tvecs[i, 1].astype(int), cp=np
        )

    return shifted_mov


def apply_plane_shiftd4D(mov, tvecs):
    """
    Applies planeshifts over the Z axis to allign a padded movie, for each time frame

    Parameters
    ----------
    mov : ndarray (nz, nt, ny, nx)
        A padded 4D array
    tvecs : ndarray
        y/x pixel shift values for each z-plane

    Returns
    -------
    shifted_mov : ndarray
        Shifted version on the input movie
    """
    nt = mov.shape[1]

    ShiftedMov = np.zeros_like(mov)
    for i in range(nt):
        ShiftedMov[:, i, :, :] = apply_plane_shifts3D(
            mov[
                :,
                i,
                :,
                :,
            ],
            tvecs,
        )

    return np.asarray(ShiftedMov)


##########################################
# reference image function cpu and gpu
def get_reference_img_gpu(
    frames,
    percent_contribute,
    niter,
    rmins=None,
    rmaxs=None,
    batch_size=20,
    max_reg_xy=50,
    sigma=[1.45, 0],
):
    """
    This function creates a reference image using the GPU for a speed up

    Parameters
    ----------
    frames : ndarray (nz, nt, ny, nx)
        The frames of the movies used to create the refernce image
    percent_contribute : float
        The percentage of input frames which contributed to the final reference image
    niter : int
        The number of interation used to create the reference image
    rmins : ndarray, optional
        The minimum values of each plane, by default None
    rmaxs : ndarray, optional
        The maximum values of each plane, by default None
    batch_size : int, optional
        The amount of frames sent to the GPU in one batch, by default 20
    max_reg_xy : int, optional
        The maximum allowed x/y shift, by default 50
    sigma : list, optional
        The smoothing values in x/y and z, by default [1.45, 0]

    Returns
    -------
    ref_image : ndarray (nz, ny, nx)
        The calcualte reference image
    ymax : ndarray (niter, nz, nt)
        The y shift for each iteration, plane andtime frame
    xmax : ndarray (niter, nz, nt)
        The x shift for each iteration, plane andtime frame
    cmax : ndarray (niter, nz, nt)
        The c shift for each iteration, plane andtime frame
    used_frames : nested list
        The first dimension is iteration number, the second is plane number. The value is the frames
        used for the reference iamge in that iteration and plane.
    """
    ncc = max_reg_xy * 2 + 1

    # get a intial guess of the reference image
    refImg = init_ref_3d(frames)

    nz, ny, nx = refImg.shape
    # Get the computed masks, once.. as its the same for thesame shape of array.. save time...
    # mask_offset need tou pdate to refimg, so call seeratley
    mult_mask, add_mask = compute_masks3D(refImg, sigma)

    # set up to keep these values
    cmax = np.zeros((niter, frames.shape[0], frames.shape[1]))
    ymax = np.zeros((niter, frames.shape[0], frames.shape[1]))
    xmax = np.zeros((niter, frames.shape[0], frames.shape[1]))
    used_frames = []

    # Allows rmins/rmaxs to be None, need as list of None
    if rmins is None and rmaxs is None:
        rmins = [None for i in range(nz)]
        rmaxs = [None for i in range(nz)]
    # how many frames contribute to the reference image per iteration
    nframes = np.linspace(20, percent_contribute * frames.shape[1], niter, dtype=np.int16)
    # init_ref function use top 20, keeping that as the start

    # GPU only, used to calculate batches for the gpu
    nt = frames.shape[1]
    batch_size = 20  # number of time steps sent to the gpu at once #param?
    nBatches = np.ceil(nt / batch_size).astype(int)

    for iter in range(niter):
        iter_frames = []
        # only recalculate mask offset when encessary
        if iter != 0:
            add_mask = compute_mask_offset(frames, mult_mask)

        # temporary
        refs_f = n.zeros((nz, ny, nx), dtype=complex)
        for z in range(
            nz
        ):  # need to go over each frame for the reference.. do sigz = 0 as well for this..
            refs_f[z] = phasecorr_ref(refImg[z, :, :].squeeze(), smooth_sigma=1.15)

        # mask is applied in rigid_2d_reg_gpu
        # need to send to gpu a few ~ 20 frames at a a time. gpu Ram
        for i in range(nBatches):
            idx1 = min(
                frames.shape[1], (i + 1) * batch_size
            )  # catch the case when the final idx would be larger than the array
            tmp_frames, tmp_ymax, tmp_xmax, tmp_cmax = reg.rigid_2d_reg_gpu(
                frames[:, i * batch_size : idx1, :, :],
                mult_mask,
                add_mask,
                refs_f,
                max_reg_xy,
                rmins,
                rmaxs,
                crosstalk_coeff=None,
                shift=True,
            )  #  param rmins/rmaxs

            # moving data from gpu to cpu
            tmp_frames = tmp_frames.swapaxes(0, 1)
            frames[:, i * batch_size : idx1, :, :] = cp.asnumpy(tmp_frames)
            ymax[iter, :, i * batch_size : idx1] = cp.asnumpy(tmp_ymax)
            xmax[iter, :, i * batch_size : idx1] = cp.asnumpy(tmp_xmax)
            cmax[iter, :, i * batch_size : idx1] = cp.asnumpy(tmp_cmax)
            del tmp_frames
            del tmp_ymax
            del tmp_xmax
            del tmp_cmax

        # nmax = max(2, int(frames.shape[1] * (1 + iter) / (2*niter) )) #original suite 2p
        nmax = nframes[iter]  # change so it uses more frames

        for z in range(nz):
            isort = np.argsort(-cmax[iter, z, :])[1:nmax]
            iter_frames.append(isort)

            refImg[z] = (
                frames[z, isort.squeeze(), :, :].mean(axis=0).squeeze()
            )  # refImg is mean of the most correlated frames
            refImg[z, :, :] = reg.shift_frame(
                refImg[z, :, :],
                int(n.round(+ymax[iter, z, isort].mean())),
                int(n.round(-xmax[iter, z, isort].mean())),
                cp=n,
            )  # ~recenter refImg

        used_frames.append(iter_frames)
    return refImg, ymax, xmax, cmax, used_frames


def get_reference_img_cpu(
    frames,
    percent_contribute,
    niter,
    rmins=None,
    rmaxs=None,
    max_reg_xy=30,
    sigma=[1.45, 0],
):
    """
    This function creates a reference image using the GPU for a speed up

    Parameters
    ----------
    frames : ndarray (nz, nt, ny, nx)
        The frames of the movies used to create the refernce image
    percent_contribute : float
        The percentage of input frames which contributed to the final reference image
    niter : int
        The number of interation used to create the reference image
    rmins : ndarray, optional
        The minimum values of each plane, by default None
    rmaxs : ndarray, optional
        The maximum values of each plane, by default None
    max_reg_xy : int, optional
        The maximum allowed x/y shift, by default 30
    sigma : list, optional
        The smoothing values in x/y and z, by default [1.45, 0]

    Returns
    -------
    ref_image : ndarray (nz, ny, nx)
        The calcualte reference image
    ymax : ndarray (niter, nz, nt)
        The y shift for each iteration, plane andtime frame
    xmax : ndarray (niter, nz, nt)
        The x shift for each iteration, plane andtime frame
    cmax : ndarray (niter, nz, nt)
        The c shift for each iteration, plane andtime frame
    used_frames : nested list
        The first dimension is iteration number, the second is plane number. The value is the frames
        used for the reference iamge in that iteration and plane.
    """
    ncc = max_reg_xy * 2 + 1 # unused?

    refImg = init_ref_3d(frames)
    nz, ny, nx = refImg.shape

    refImg = init_ref_3d(frames)
    nz, ny, nx = refImg.shape

    # Allows rmins/rmaxs to be None, need as list of None
    if rmins is None and rmaxs is None:
        rmins = [None for i in range(nz)]
        rmaxs = [None for i in range(nz)]

    # Get the computed masks, once.. as its the same for thesame shape of array.. save time...
    # mask_offset need toupdate to refimg, so cal seeratley
    mult_mask, add_mask = compute_masks3D(refImg, sigma)

    # create empty array for values to keep
    cmax = np.zeros((niter, frames.shape[0], frames.shape[1]))
    ymax = np.zeros((niter, frames.shape[0], frames.shape[1]))
    xmax = np.zeros((niter, frames.shape[0], frames.shape[1]))
    used_frames = []

    nframes = np.linspace(20, percent_contribute * frames.shape[1], niter, dtype=np.int16)
    for iter in range(niter):
        iter_frames = []
        # only get offset mask if needed
        if iter != 0:
            add_mask = compute_mask_offset(frames, mult_mask)

        # temporary
        refs_f = n.zeros((nz, ny, nx))

        for z in range(
            nz
        ):  # need to go iver each frame for the reference.. do sigz = 0 as well for this..
            refs_f[z] = phasecorr_ref(refImg[z, :, :].squeeze(), smooth_sigma=1.15)

        # mask is applied in rigid_2d_reg_cpu
        # this will only return mov_shifted, ymaxs, xmaxs
        # tmp, ymax[iter], xmax[iter], cmax[iter] = reg.rigid_2d_reg_cpu(
        tmp, ymax[iter], xmax[iter] = reg.rigid_2d_reg_cpu(
            frames,
            mult_mask,
            add_mask,
            refs_f,
            max_reg_xy,
            rmins,
            rmaxs,
            crosstalk_coeff=None,
            shift=True,
        )  # check max_reg_xy
        frames = tmp.swapaxes(
            0, 1
        )  # need to swap axes as by default reigid_2d_reg_cpu return nt,nz not nz,nt
        del tmp

        nmax = nframes[iter]  # pick amount of frames which contribute
        for z in range(nz):
            isort = n.argsort(-cmax[iter, z, :])[1:nmax]

            refImg[z] = (
                frames[z, isort.squeeze(), :, :].mean(axis=0).squeeze()
            )  # refImg is mean of the most correlated frames
            refImg[z, :, :] = reg.shift_frame(
                refImg[z, :, :],
                int(n.round(+ymax[iter, z, isort].mean())),
                int(n.round(-xmax[iter, z, isort].mean())),
                cp=n,
            )  # ~recenter refImg
        used_frames.append(iter_frames)
    return refImg, ymax, xmax, cmax, used_frames


############################################################
# New fuse_mov and pad_mov function to allow fuse/pad only
def fuse_mov(mov, fuse_shift, xs):
    """
    This function fuses the movie in the x axis to account for how the movie is stored

    Parameters
    ----------
    mov : ndarray (nz, nt, ny, nx)
        Unfused raw movie
    fuse_shift : int
        number of x pixels which needs to be fused
    xs : ndarray
        The starting points of x-strips of the movie

    Returns
    -------
    mov_fuse : ndarray (nz, nt, ny, new nx)
        Fused version of the movie
    new_xs : list
        The new starting/end points of the fused movie x-strips
    og_xs : list
        The original starting/end points of the fused movie x-strips
    """
    nz, nt, nyo, nxo = mov.shape
    n_stitches = len(xs) - 1
    n_xpix_lost_for_fusing = n_stitches * fuse_shift

    # new size of mov after fusing x pixels
    nyn = nyo
    nxn = nxo - n_xpix_lost_for_fusing

    mov_fuse = n.zeros((nz, nt, nyn, nxn), n.float32)

    lshift = fuse_shift // 2
    rshift = fuse_shift - lshift
    xn0 = 0
    og_xs = []
    new_xs = []
    for i in range(n_stitches + 1):
        x0 = xs[i]
        if i > 0:
            x0 += lshift
        if i == n_stitches:
            x1 = nxo
        else:
            x1 = xs[i + 1] - rshift
        dx = x1 - x0

        mov_fuse[:, :, :nyo, xn0 : xn0 + dx] = mov[:, :, :, x0:x1]
        new_xs.append((xn0, xn0 + dx))
        og_xs.append((x0, x1))
        xn0 += dx
    return mov_fuse, new_xs, og_xs


def pad_mov(mov, plane_shifts):
    """
    This function pads the movie as to allow inter z-plane shifts to allign the z-planes of the movie

    Parameters
    ----------
    mov : ndarray (nz, nt, ny, nx)
        The movie to be padded
    plane_shifts : ndarray (nz,2)
        The y/x shifts for each z-plane to allign the movie

    Returns
    -------
    mov_pad : ndarray (nz, nt, new ny, new nx)
        The padded movie
    xpad :
        The amount of x pixels padded
    ypad :
        The amount of y pixels padded
    """
    nz, nt, nyo, nxo = mov.shape

    plane_shifts = n.round(plane_shifts).astype(int)

    xrange = plane_shifts[:, 1].min(), plane_shifts[:, 1].max()
    yrange = plane_shifts[:, 0].min(), plane_shifts[:, 0].max()

    ypad = n.ceil(n.abs(n.diff(yrange))).astype(int)[::-1]
    yshift = n.ceil(n.abs((yrange[0]))).astype(int)
    xpad = n.ceil(n.abs(n.diff(xrange))).astype(int)[::-1]
    xshift = n.ceil(n.abs((xrange[0]))).astype(int)
    nyn = nyo + ypad.sum()
    nxn = nxo + xpad.sum()

    mov_pad = n.zeros((nz, nt, nyn, nxn), n.float32)

    mov_pad[:, :, yshift : yshift + nyo, xshift : xshift + nxo] = mov[:]
    return mov_pad, xpad, ypad


def pad_mov3D(mov, plane_shifts):
    """
    This function pads the movie as to allow inter z-plane shifts to allign the z-planes of the movie

    Parameters
    ----------
    mov : ndarray (nz, ny, nx)
        The movie to be padded
    plane_shifts : ndarray (nz,2)
        The y/x shifts for each z-plane to allign the movie

    Returns
    -------
    mov_pad : ndarray (nz, nt, new ny, new nx)
        The padded movie
    xpad :
        The amount of x pixels padded
    ypad :
        The amount of y pixels padded
    """
    nz, nyo, nxo = mov.shape

    plane_shifts = n.round(plane_shifts).astype(int)

    xrange = plane_shifts[:, 1].min(), plane_shifts[:, 1].max()
    yrange = plane_shifts[:, 0].min(), plane_shifts[:, 0].max()

    ypad = n.ceil(n.abs(n.diff(yrange))).astype(int)[::-1]
    yshift = n.ceil(n.abs((yrange[0]))).astype(int)
    xpad = n.ceil(n.abs(n.diff(xrange))).astype(int)[::-1]
    xshift = n.ceil(n.abs((xrange[0]))).astype(int)
    nyn = nyo + ypad.sum()
    nxn = nxo + xpad.sum()

    mov_pad = n.zeros((nz, nyn, nxn), n.float32)

    mov_pad[:, yshift : yshift + nyo, xshift : xshift + nxo] = mov[:]
    return mov_pad, xpad, ypad


##############################################################################################
# Function took/adapted from suite 2p for creating nonrigid masks
def calculate_nblocks(N, block_size):
    """
    Returns block_size and nblocks from dimension length and desired block size
    Took/adapted from suite2P

    Parameters
    ----------
    N: int
        The total size of the blocked dimension
    block_size: int
        The size of each block

    Returns
    -------
    block_size: int
    nblocks: int
    """
    return (N, 1) if block_size >= N else (block_size, int(np.ceil(1.5 * N / block_size)))


def kernelD2(xs: int, ys: int):
    """
    Parameters
    Took/adapted from suite2P
    ----------
    xs: ndarray
        np.arange(No of x blocks)
    ys: ndarray
        np.arange(No of y blocks)

    Returns
    -------
        Upsampling kernel
    """
    ys, xs = np.meshgrid(xs, ys)
    ys = ys.flatten().reshape(1, -1)
    xs = xs.flatten().reshape(1, -1)
    R = np.exp(-((ys - ys.T) ** 2 + (xs - xs.T) ** 2))
    R = R / np.sum(R, axis=0)
    return R


def spatial_taper(sig, Ly, Lx):
    """
    Returns spatial taper  on edges with gaussian of std sig
    Took/adapted from suite2P

    Parameters
    ----------
    sig : float
        Standard deviation of the spatial taper width
    Ly: int
        frame height
    Lx: int
        frame width

    Returns
    -------
    maskMul
    """
    xx, yy = meshgrid_mean_centered(x=Lx, y=Ly)
    mY = ((Ly - 1) / 2) - 2 * sig
    mX = ((Lx - 1) / 2) - 2 * sig
    maskY = 1.0 / (1.0 + np.exp((yy - mY) / sig))
    maskX = 1.0 / (1.0 + np.exp((xx - mX) / sig))
    maskMul = maskY * maskX
    return maskMul


def make_blocks(Ly, Lx, block_size=(128, 128)):
    """
    Computes overlapping blocks to split FOV into to register separately
    Took/adapted from suite2P

    Parameters
    ----------
    Ly: int
        Number of pixels in the vertical dimension
    Lx: int
        Number of pixels in the horizontal dimension
    block_size: int, int
        block size (No of y blocks, No of x blocks)

    Returns
    -------
    yblock: float array
    xblock: float array
    nblocks: int, int
    block_size: int, int
    NRsm: array
    """

    block_size_y, ny = calculate_nblocks(N=Ly, block_size=block_size[0])
    block_size_x, nx = calculate_nblocks(N=Lx, block_size=block_size[1])
    block_size = (block_size_y, block_size_x)

    # todo: could rounding to int here over-represent some pixels over others? - suite2P
    ystart = np.linspace(0, Ly - block_size[0], ny).astype("int")
    xstart = np.linspace(0, Lx - block_size[1], nx).astype("int")
    yblock = [
        np.array([ystart[iy], ystart[iy] + block_size[0]])
        for iy in range(ny)
        for _ in range(nx)
    ]
    xblock = [
        np.array([xstart[ix], xstart[ix] + block_size[1]])
        for _ in range(ny)
        for ix in range(nx)
    ]

    NRsm = kernelD2(xs=np.arange(nx), ys=np.arange(ny)).T

    return yblock, xblock, [ny, nx], block_size, NRsm


def phasecorr_reference(refImg0, maskSlope, smooth_sigma, yblock, xblock):
    """
    Computes taper and fft'ed reference image for phasecorr, non rigid case.
    Took/adapted from suite2P

    Parameters
    ----------
    refImg0: array
    maskSlope
    smooth_sigma
    yblock: float array
    xblock: float array

    Returns
    -------
    maskMul
    maskOffset
    cfRefImg

    """
    nb, Ly, Lx = len(yblock), yblock[0][1] - yblock[0][0], xblock[0][1] - xblock[0][0]
    dims = (nb, Ly, Lx)
    cfRef_dims = dims
    gaussian_filter = gaussian_fft(smooth_sigma, *cfRef_dims[1:])
    cfRefImg1 = np.empty(cfRef_dims, "complex64")

    maskMul = spatial_taper(maskSlope, *refImg0.shape)
    maskMul1 = np.empty(dims, "float32")
    maskMul1[:] = spatial_taper(2 * smooth_sigma, Ly, Lx)
    maskOffset1 = np.empty(dims, "float32")
    for yind, xind, maskMul1_n, maskOffset1_n, cfRefImg1_n in zip(
        yblock, xblock, maskMul1, maskOffset1, cfRefImg1
    ):
        ix = np.ix_(
            np.arange(yind[0], yind[-1]).astype("int"),
            np.arange(xind[0], xind[-1]).astype("int"),
        )
        refImg = refImg0[ix]

        # mask params
        maskMul1_n *= maskMul[ix]
        maskOffset1_n[:] = refImg.mean() * (1.0 - maskMul1_n)

        # gaussian filter
        cfRefImg1_n[:] = np.conj(fft.fft2(refImg))
        cfRefImg1_n /= 1e-5 + np.absolute(cfRefImg1_n)
        cfRefImg1_n[:] *= gaussian_filter

    return (
        maskMul1[:, np.newaxis, :, :],
        maskOffset1[:, np.newaxis, :, :],
        cfRefImg1[:, np.newaxis, :, :],
    )


# NEW functions for 3d regsitration
def compute_reference_and_masks_3d(
    mov_cpu, reference_params, log_cb=default_log, rmins=None, rmaxs=None, use_GPU=True
):
    percent_contribute = reference_params["percent_contribute"]
    niter = reference_params["niter"]
    sigma = reference_params["sigma"]
    pc_size = reference_params[
        "pc_size"
    ]  # NOTEadd to reference params, this is the 3d version of max_reg_xy

    nz, nt, ny, nx = mov_cpu.shape

    if use_GPU:
        batch_size = reference_params["batch_size"]

    # need to do plane_shifts first in 3D case
    log_cb("Computing plane alignment shifts", 1)
    tvecs = align_planes(mov_cpu.mean(axis=1), reference_params)

    # correct bad tvec estimates
    # print(reference_params)
    # print(tvecs)
    if reference_params.get("fix_shallow_plane_shift_estimates", False):
        shallow_plane_thresh = reference_params.get(
            "fix_shallow_plane_shift_esimate_threshold", 20
        )
        peaks = n.abs(tvecs[:shallow_plane_thresh]).max(axis=0)
        bad_planes = n.logical_or(
            n.abs(tvecs[shallow_plane_thresh:, 0]) > peaks[0],
            n.abs(tvecs[shallow_plane_thresh:, 1]) > peaks[1],
        )
        tvecs[shallow_plane_thresh:][bad_planes, :] = 0
        if bad_planes.sum() > 0:
            log_cb("Fixing %d plane alignment outliers" % bad_planes.sum(), 2)

    mov_cpu, xpad, ypad = pad_mov(mov_cpu, tvecs)  # pad the movie to correct size
    pad_sizes = [xpad, ypad]
    xpad = int(xpad)
    ypad = int(ypad)

    log_cb("Applying plane alignment shifts", 1)
    # print(mov_cpu.shape)
    # print(tvecs.shape)
    # print(tvecs)
    mov_cpu = reg_3d.shift_mov_lbm_fast(mov_cpu, tvecs)  # apply the lbm shift

    if use_GPU:
        log_cb("Launching 3D GPU reference image calculation", 1)
        ref_img, zmax, ymax, xmax, cmax, used_frames, subpix_shifts, shifted_mov = (
            get_reference_img_gpu_3d(
                mov_cpu,
                percent_contribute,
                niter,
                xpad,
                ypad,
                rmins=rmins,
                rmaxs=rmaxs,
                batch_size=batch_size,
                pc_size=pc_size,
                sigma=sigma,
                log_cb=log_cb,
            )
        )
    else:
        log_cb("Launching 3D CPU reference image calculation", 1)
        ref_img, ymax, xmax, cmax, used_frames = get_reference_img_cpu_3d(
            mov_cpu,
            percent_contribute,
            niter,
            xpad,
            ypad,
            rmins=rmins,
            rmaxs=rmaxs,
            pc_size=pc_size,
            sigma=sigma,
        )
        zmax = None
        subpix_shifts = (
            None  # TODO transfer changes Ali made to use_GPU option to to CPU version
        )
        shifted_mov = None  # TODO also return the shifted version of mov_cpu in this version, like the GPU version

    # Option to clip the ref_image per plane for below
    plane_mins = np.zeros(nz)
    plane_maxs = np.zeros(nz)
    clipped_ref_img = np.zeros_like(ref_img)
    for z in range(nz):
        rmin, rmax = n.int16(n.percentile(ref_img[z], 1)), n.int16(
            n.percentile(ref_img[z], 99)
        )
        clipped_ref_img[z] = n.clip(ref_img[z], rmin, rmax)
        plane_mins[z], plane_maxs[z] = rmin, rmax

    # add useful info to reference results
    reference_info = {}
    reference_info["plane_mins"] = plane_mins
    reference_info["plane_maxs"] = plane_maxs
    reference_info["zmax"] = zmax
    reference_info["ymax"] = ymax
    reference_info["xmax"] = xmax
    reference_info["cmax"] = cmax
    reference_info["subpix_shifts"] = subpix_shifts
    reference_info["used_frames"] = used_frames

    frame_use_order = n.zeros((nt))
    frame_use_order[:] = niter
    for i in range(nt):
        for j in range(niter):
            if i in used_frames[j]:
                frame_use_order[i] = j
                break
    reference_info["frame_use_order"] = frame_use_order

    if reference_params.get("norm_frames", True):
        all_refs_and_masks, reference_params = get_phasecorr_and_masks(
            clipped_ref_img, reference_params
        )
    else:
        all_refs_and_masks, reference_params = get_phasecorr_and_masks(
            ref_img, reference_params
        )

    return (
        tvecs,
        ref_img,
        all_refs_and_masks,
        pad_sizes,
        reference_params,
        reference_info,
        shifted_mov,
    )


def get_reference_img_gpu_3d(
    mov_cpu,
    percent_contribute,
    niter,
    xpad,
    ypad,
    rmins=None,
    rmaxs=None,
    batch_size=20,
    pc_size=(2, 20, 20),
    sigma=(0, 1.5),
    log_cb=default_log,
):
    # log_cb("Launched")
    # print(xpad, ypad)
    if ypad == 0:
        if xpad == 0:
            mov_cropped = mov_cpu.copy()
        else:
            mov_cropped = mov_cpu[:, :, xpad:-xpad]
    elif xpad == 0:
        mov_cropped = mov_cpu[:, ypad:-ypad]
    else:
        mov_cropped = mov_cpu[:, :, ypad:-ypad, xpad:-xpad]

    # print(mov_cpu.shape)
    use_most_correlated_frames = False

    if use_most_correlated_frames:
        # TODO this takes very long for a volume with 400 frames
        tic = time.time()
        log_cb("Seeding reference image with most correlated frames", 2)
        ref_img = init_ref_3d(mov_cropped)
        log_cb(f"Seeded in {time.time() - tic : .2f} s", 2)
    else:
        log_cb("Seeding reference image with most active frames", 2)
        # print(mov_cropped.shape)
        # this is a hack to take the 20 frames with the most activity
        mean_activity_per_frame = mov_cropped.mean(axis=(0, 2, 3))
        top_frames = n.argsort(mean_activity_per_frame)[-20:]
        ref_img = mov_cropped[:, top_frames].mean(axis=1)

    mult_mask, add_mask = compute_masks3D(ref_img, sigma)

    cmax = np.zeros((niter, mov_cropped.shape[1]))
    ymax = np.zeros((niter, mov_cropped.shape[1]))
    xmax = np.zeros((niter, mov_cropped.shape[1]))
    zmax = np.zeros((niter, mov_cropped.shape[1]))
    subpix_shifts = np.zeros((niter, mov_cropped.shape[1], 3))

    used_frames = []

    # start at 20 (no frames used for init_ref) + 10% of nt (so add some frame on the first iteration) and finish at percent_contribute * nt
    n_frames = np.linspace(
        20 + mov_cropped.shape[1] * 0.1,
        percent_contribute * mov_cropped.shape[1],
        niter,
        dtype=np.int16,
    )

    nt = mov_cropped.shape[1]

    tic = time.time()
    for iter_idx in range(niter):
        if iter_idx != 0:
            add_mask = compute_mask_offset(ref_img, mult_mask)

        refs_f = reg_3d.mask_filter_fft_ref(ref_img, mult_mask, add_mask, smooth=0.5)

        phase_corr_shifted, int_shift, pc_peak_loc, subpix_shift, __ = (
            reg_3d.rigid_3d_ref_gpu(
                mov_cropped,
                mult_mask,
                add_mask,
                refs_f,
                pc_size,
                batch_size=batch_size,
                rmins=None,
                rmaxs=None,
                crosstalk_coeff=None,
            )
        )
        pc_peak_loc = pc_peak_loc.astype(np.int32)
        int_shift = int_shift.astype(np.int32)

        for t in range(nt):
            cmax[iter_idx, t] = phase_corr_shifted[
                t, pc_peak_loc[t, 0], pc_peak_loc[t, 1], pc_peak_loc[t, 2]
            ]
            xmax[iter_idx, t] = int_shift[t, 2]
            ymax[iter_idx, t] = int_shift[t, 1]
            zmax[iter_idx, t] = int_shift[t, 0]
            subpix_shifts[iter_idx, t] = subpix_shift[t]
        nmax = n_frames[iter_idx]

        isort = np.argsort(-cmax[iter_idx, :])[1:nmax]
        used_frames.append(isort)

        if iter_idx != (
            niter - 1
        ):  # for the last iteration dont need to remake the reference on the subset
            # NOTE should there be a -sign here
            shifted_img_iter = reg_3d.shift_mov_fast(
                mov_cropped[:, isort, :, :], -int_shift[isort, :]
            )
            ref_img = shifted_img_iter.mean(axis=1)
            # recenter img
            ref_img = reg_3d.shift_mov_fast(
                ref_img[:, np.newaxis, :, :],
                int_shift[isort, :].mean(axis=0)[np.newaxis, :].astype(np.int32),
            ).squeeze()
        toc = time.time() - tic
        tic = time.time()
        log_cb(
            f"Completed iter {iter_idx+1} out of {niter} in {toc: .2f}s using {len(isort): 03d}/{nt} frames",
            2,
        )

    # #create the uncropped reference img, after all the iterations!
    # shifted_img = reg_3d.shift_mov_fast(mov_cpu[:,isort,:,:].copy(), -int_shift[isort,:])
    # full_ref_im = shifted_img.mean(axis = 1)
    # full_ref_im = reg_3d.shift_mov_fast(full_ref_im[:,np.newaxis,:,:], int_shift[isort,:].mean(axis=0)[np.newaxis,:].astype(np.int32)).squeeze()

    # # create the full shifted mov
    # # TODO SAM: I DONT KNOW IF THESE SHIFTS ARE RIGHT - CAN YOU DOUBLE CHECK? THANKS - ALI
    # # Use the same correction shifts as the reference image as keep the shifted_mov alligned to the reference img!
    # shifted_mov = reg_3d.shift_mov_fast(mov_cpu, int_shift[isort,:].mean(axis=0)[np.newaxis,:].astype(np.int32) - int_shift)

    # do shift once, use only frames in the reference image to re-center
    # print("Shifting movie")
    shifted_mov = reg_3d.shift_mov_fast(
        mov_cpu,
        int_shift[isort, :].mean(axis=0)[np.newaxis, :].astype(np.int32) - int_shift,
    )
    # print("Shifted movie")
    full_ref_im = shifted_mov[:, isort, :, :].mean(axis=1)
    log_cb(f"Used {isort.shape[0]} frames to make the reference image", 2)
    return full_ref_im, zmax, ymax, xmax, cmax, used_frames, subpix_shifts, shifted_mov


def get_reference_img_cpu_3d(
    mov_cpu,
    percent_contribute,
    niter,
    xpad,
    ypad,
    rmins=None,
    rmaxs=None,
    pc_size=(2, 30, 30),
    sigma=[0, 1.5],
):
    if ypad == 0:
        if xpad == 0:
            mov_cropped = mov_cpu.copy()
        mov_cropped = mov_cpu[:, :, xpad:-xpad]
    elif xpad == 0:
        mov_cropped = mov_cpu[:, ypad:-ypad]
    else:
        mov_cropped = mov_cpu[:, :, ypad:-ypad, xpad:-xpad]
    ref_img = init_ref_3d(mov_cropped)

    mult_mask, add_mask = compute_masks3D(ref_img, sigma)

    cmax = np.zeros((niter, mov_cropped.shape[1]))
    ymax = np.zeros((niter, mov_cropped.shape[1]))
    xmax = np.zeros((niter, mov_cropped.shape[1]))
    used_frames = []

    # start at 20 (no frames used for init_ref) + 10% of nt (so add some frame on the first iteration) and finish at percent_contribute * nt
    n_frames = np.linspace(
        20 + mov_cropped.shape[1] * 0.1,
        percent_contribute * mov_cropped.shape[1],
        niter,
        dtype=np.int16,
    )

    nt = mov_cropped.shape[1]

    for iter in range(niter):
        if iter != 0:
            add_mask = compute_mask_offset(ref_img, mult_mask)

        refs_f = reg_3d.mask_filter_fft_ref(ref_img, mult_mask, add_mask, smooth=0.5)

        phase_corr_shifted, int_shift, pc_peak_loc, __ = reg_3d.rigid_3d_ref_cpu(
            mov_cropped,
            mult_mask,
            add_mask,
            refs_f,
            pc_size,
            rmins=None,
            rmaxs=None,
            crosstalk_coeff=None,
        )
        pc_peak_loc = pc_peak_loc.astype(np.int32)
        int_shift = int_shift.astype(np.int32)

        for t in range(nt):
            cmax[iter, t] = phase_corr_shifted[
                t, pc_peak_loc[t, 0], pc_peak_loc[t, 1], pc_peak_loc[t, 2]
            ]
            xmax[iter, t] = int_shift[t, 2]
            ymax[iter, t] = int_shift[t, 1]
        nmax = n_frames[iter]

        isort = np.argsort(-cmax[iter, :])[1:nmax]
        used_frames.append(isort)

        if iter != (
            niter - 1
        ):  # for the last iteration dont need to remake the reference on the subset
            # NOTE Check if thisshould be -
            shifted_img_iter = reg_3d.shift_mov_fast(
                mov_cropped[:, isort, :, :], -int_shift[isort, :]
            )
            ref_img = shifted_img_iter.mean(axis=1)
            # recenter img
            ref_img = reg_3d.shift_mov_fast(
                ref_img[:, np.newaxis, :, :],
                -int_shift[isort, :].mean(axis=0)[np.newaxis, :].astype(np.int32),
            ).squeeze()

    # create the uncropped reference img, after all the iterations!
    shifted_img = reg_3d.shift_mov_fast(mov_cpu[:, isort, :, :], int_shift[isort, :])
    full_ref_im = shifted_img.mean(axis=1)
    full_ref_im = reg_3d.shift_mov_fast(
        full_ref_im[:, np.newaxis, :, :],
        -int_shift[isort, :].mean(axis=0)[np.newaxis, :].astype(np.int32),
    ).squeeze()

    return full_ref_im, ymax, xmax, cmax, used_frames
