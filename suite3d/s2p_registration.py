import numpy as np
from numba import float32, njit, prange
from scipy.ndimage import gaussian_filter1d
from numba import vectorize, complex64

try:
    from mkl_fft import fft2, ifft2
except ImportError:
    from scipy.fftpack import fft2, ifft2

from functools import lru_cache

from typing import Tuple

#############################################
# 
# Code here is taken directly from Suite2p
#  Credit to Stringer & Pachitariu et al.
#   https://github.com/MouseLand/suite2p
# 
#############################################


@vectorize([complex64(complex64, complex64)], nopython=True, target='parallel')
def apply_dotnorm(Y, cfRefImg):
    return Y / (np.complex64(1e-5) + np.abs(Y)) * cfRefImg
@vectorize(['complex64(int16, float32, float32)', 'complex64(float32, float32, float32)'], nopython=True, target='parallel', cache=True)
def addmultiply(x, mul, add):
    return np.complex64(np.float32(x) * mul + add)
def convolve(mov: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Returns the 3D array 'mov' convolved by a 2D array 'img'.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to process
    img: 2D array
        The convolution kernel

    Returns
    -------
    convolved_data: nImg x Ly x Lx
    """
    return ifft2(apply_dotnorm(fft2(mov), img)) #.astype(np.complex64)

def register_frames(refAndMasks, frames, ops=None, base_shift=None,  do_rigid=True):
    """ register frames to reference image 
    
    Parameters
    ----------

    ops : dictionary or list of dicts
        'Ly', 'Lx', 'batch_size', 'align_by_chan', 'nonrigid'
        (optional 'keep_movie_raw', 'raw_file')

    refImg : 2D array (optional, default None)

    raw : bool (optional, default True)
        use raw_file for registration if available, if False forces reg_file to be used
    
    base_shift : tuple (optional, default None)
        a tuple in the form (dy, dx) of a constant shift to be applied to all frames

    Returns
    --------

    ops : dictionary
        'nframes', 'yoff', 'xoff', 'corrXY', 'yoff1', 'xoff1', 'corrXY1', 'badframes'


    """
    if len(refAndMasks)==6 or not isinstance(refAndMasks, np.ndarray):
        maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR = refAndMasks 
    else: assert False
        


    fsmooth = frames.copy().astype(np.float32)
    if ops['smooth_sigma_time'] > 0:
        fsmooth = gaussian_filter1d(data=fsmooth, sigma=ops['smooth_sigma_time'], axis=0)

    # rigid registration
    if ops.get('norm_frames', False):
        fsmooth = np.clip(fsmooth, ops['rmin'], ops['rmax'])
    
    if do_rigid:
        ymax, xmax, cmax = phasecorr(
            data=addmultiply(fsmooth,maskMul,maskOffset),
            cfRefImg=cfRefImg,
            maxregshift=ops['maxregshift'],
            smooth_sigma_time=ops['smooth_sigma_time'], 
        )
        if base_shift is not None:
            if ops['nonrigid']: print("base_shift with nonrigid on is broken!")
            ymax += base_shift[0]
            xmax += base_shift[1]

        for frame, dy, dx in zip(frames, ymax, xmax):
            frame[:] = shift_frame(frame=frame, dy=dy, dx=dx)
    else:
        ymax = None; xmax = None; cmax = None;

    # non-rigid registration
    if ops['nonrigid']:
        # need to also shift smoothed data (if smoothing used)
        if ops['smooth_sigma_time']:
            for fsm, dy, dx in zip(fsmooth, ymax, xmax):
                fsm[:] = shift_frame(frame=fsm, dy=dy, dx=dx)
        else:
            fsmooth = frames.copy()

        if ops.get('norm_frames', False):
            fsmooth = np.clip(fsmooth, ops['rmin'], ops['rmax'])
            
        ymax1, xmax1, cmax1 =nr_phasecorr(
            data=fsmooth,
            maskMul=maskMulNR.squeeze(),
            maskOffset=maskOffsetNR.squeeze(),
            cfRefImg=cfRefImgNR.squeeze(),
            snr_thresh=ops['snr_thresh'],
            NRsm=ops['NRsm'],
            xblock=ops['xblock'],
            yblock=ops['yblock'],
            maxregshiftNR=ops['maxregshiftNR'],
        )

        frames = nonrigid_transform_data(
            data=frames,
            nblocks=ops['nblocks'],
            xblock=ops['xblock'],
            yblock=ops['yblock'],
            ymax1=ymax1,
            xmax1=xmax1,
        )
    else:
        ymax1, xmax1, cmax1 = None, None, None 
    
    return frames, ymax, xmax, cmax, ymax1, xmax1, cmax1

def phasecorr(data, cfRefImg, maxregshift, smooth_sigma_time) -> Tuple[int, int, float]:
    """ compute phase correlation between data and reference image

    Parameters
    ----------
    data : int16
        array that's frames x Ly x Lx
    maxregshift : float
        maximum shift as a fraction of the minimum dimension of data (min(Ly,Lx) * maxregshift)
    smooth_sigma_time : float
        how many frames to smooth in time

    Returns
    -------
    ymax : int
        shifts in y from cfRefImg to data for each frame
    xmax : int
        shifts in x from cfRefImg to data for each frame
    cmax : float
        maximum of phase correlation for each frame

    """
    min_dim = np.minimum(*data.shape[1:])  # maximum registration shift allowed
    lcorr = int(np.minimum(np.round(maxregshift * min_dim), min_dim // 2))
    
    data = convolve(data, cfRefImg)

    cc = np.real(
            np.block(
                [[data[:,  -lcorr:, -lcorr:], data[:,  -lcorr:, :lcorr+1]],
                [data[:, :lcorr+1, -lcorr:], data[:, :lcorr+1, :lcorr+1]]]
            )
        )
    
    cc = gaussian_filter1d(cc, smooth_sigma_time,axis=0) if smooth_sigma_time > 0 else cc

    ymax, xmax = np.zeros(data.shape[0], np.int32), np.zeros(data.shape[0], np.int32)
    for t in np.arange(data.shape[0]):
        ymax[t], xmax[t] = np.unravel_index(np.argmax(cc[t], axis=None), (2 * lcorr + 1, 2 * lcorr + 1))
    cmax = cc[np.arange(len(cc)), ymax, xmax]
    ymax, xmax = ymax - lcorr, xmax - lcorr

    return ymax, xmax, cmax.astype(np.float32)






@njit(['(int16[:, :],float32[:,:], float32[:,:], float32[:,:])', 
        '(float32[:, :],float32[:,:], float32[:,:], float32[:,:])'], cache=True)
def map_coordinates(I, yc, xc, Y) -> None:
    """
    In-place bilinear transform of image 'I' with ycoordinates yc and xcoordinates xc to Y
    
    Parameters
    -------------
    I : Ly x Lx
    yc : Ly x Lx
        new y coordinates
    xc : Ly x Lx
        new x coordinates
    Y : Ly x Lx
        shifted I
    """
    Ly,Lx = I.shape
    yc_floor = yc.astype(np.int32)
    xc_floor = xc.astype(np.int32)
    yc = yc - yc_floor
    xc = xc - xc_floor
    for i in range(yc_floor.shape[0]):
        for j in range(yc_floor.shape[1]):
            yf = min(Ly-1, max(0, yc_floor[i,j]))
            xf = min(Lx-1, max(0, xc_floor[i,j]))
            yf1= min(Ly-1, yf+1)
            xf1= min(Lx-1, xf+1)
            y = yc[i,j]
            x = xc[i,j]
            Y[i,j] = (np.float32(I[yf, xf]) * (1 - y) * (1 - x) +
                      np.float32(I[yf, xf1]) * (1 - y) * x +
                      np.float32(I[yf1, xf]) * y * (1 - x) +
                      np.float32(I[yf1, xf1]) * y * x )

@njit(['int16[:, :,:], float32[:,:,:], float32[:,:,:], float32[:,:], float32[:,:], float32[:,:,:]',
       'float32[:, :,:], float32[:,:,:], float32[:,:,:], float32[:,:], float32[:,:], float32[:,:,:]'], parallel=True, cache=True)
def shift_coordinates(data, yup, xup, mshy, mshx, Y):
    """
    Shift data into yup and xup coordinates

    Parameters
    ----------
    data : nimg x Ly x Lx
    yup : nimg x Ly x Lx
        y shifts for each coordinate
    xup : nimg x Ly x Lx
        x shifts for each coordinate
    mshy : Ly x Lx
        meshgrid in y
    mshx : Ly x Lx
        meshgrid in x
    Y : nimg x Ly x Lx
        shifted data
    """
    for t in prange(data.shape[0]):
        map_coordinates(data[t], mshy+yup[t], mshx+xup[t], Y[t])


@njit((float32[:, :,:], float32[:,:,:], float32[:,:], float32[:,:], float32[:,:,:], float32[:,:,:]), parallel=True, cache=True)
def block_interp(ymax1, xmax1, mshy, mshx, yup, xup):
    """
    interpolate from ymax1 to mshy to create coordinate transforms

    Parameters
    ----------
    ymax1
    xmax1
    mshy: Ly x Lx
        meshgrid in y
    mshx: Ly x Lx
        meshgrid in x
    yup: nimg x Ly x Lx
        y shifts for each coordinate
    xup: nimg x Ly x Lx
        x shifts for each coordinate
    """
    for t in prange(ymax1.shape[0]):
        map_coordinates(ymax1[t], mshy, mshx, yup[t])  # y shifts for blocks to coordinate map
        map_coordinates(xmax1[t], mshy, mshx, xup[t])  # x shifts for blocks to coordinate map


def upsample_block_shifts(Lx, Ly, nblocks, xblock, yblock, ymax1, xmax1):
    """ upsample blocks of shifts into full pixel-wise maps for shifting

    this function upsamples ymax1, xmax1 so that they are nimg x Ly x Lx
    for later bilinear interpolation
        

    Parameters
    ----------
    Lx: int
        number of pixels in the horizontal dimension
    Ly: int
        number of pixels in the vertical dimension
    nblocks: (int, int)
    xblock: float array
    yblock: float array
    ymax1: nimg x nblocks
        y shifts of blocks
    xmax1: nimg x nblocks
        y shifts of blocks

    Returns
    -------
    yup : nimg x Ly x Lx
        y shifts for each coordinate
    xup : nimg x Ly x Lx
        x shifts for each coordinate

    """
    # make arrays of control points for piecewise-affine transform
    # includes centers of blocks AND edges of blocks
    # note indices are flipped for control points
    # block centers
    yb = np.array(yblock[::nblocks[1]]).mean(axis=1)  # this recovers the coordinates of the meshgrid from (yblock, xblock)
    xb = np.array(xblock[:nblocks[1]]).mean(axis=1)

    iy = np.interp(np.arange(Ly), yb, np.arange(yb.size)).astype(np.float32)
    ix = np.interp(np.arange(Lx), xb, np.arange(xb.size)).astype(np.float32)
    mshx, mshy = np.meshgrid(ix, iy)

    # interpolate from block centers to all points Ly x Lx
    nimg = ymax1.shape[0]
    ymax1 = ymax1.reshape(nimg, nblocks[0], nblocks[1])
    xmax1 = xmax1.reshape(nimg, nblocks[0], nblocks[1])
    yup = np.zeros((nimg, Ly, Lx), np.float32)
    xup = np.zeros((nimg, Ly, Lx), np.float32)

    block_interp(ymax1, xmax1, mshy, mshx, yup, xup)

    return yup, xup


def nonrigid_transform_data(data, nblocks, xblock, yblock, ymax1, xmax1, bilinear=True):
    """
    Piecewise affine transformation of data using block shifts ymax1, xmax1
    
    Parameters
    ----------

    data : nimg x Ly x Lx
    nblocks: (int, int)
    xblock: float array
    yblock: float array
    ymax1 : nimg x nblocks
        y shifts of blocks
    xmax1 : nimg x nblocks
        y shifts of blocks
    bilinear: bool (optional, default=True)
        do bilinear interpolation, if False do nearest neighbor

    Returns
    -----------
    Y : float32, nimg x Ly x Lx
        shifted data
    """
    _, Ly, Lx = data.shape
    yup, xup = upsample_block_shifts(
        Lx=Lx,
        Ly=Ly,
        nblocks=nblocks,
        xblock=xblock,
        yblock=yblock,
        ymax1=ymax1,
        xmax1=xmax1,
    )
    if not bilinear:
        yup = np.round(yup)
        xup = np.round(xup)

    # use shifts and do bilinear interpolation
    mshx, mshy = np.meshgrid(np.arange(Lx, dtype=np.float32), np.arange(Ly, dtype=np.float32))
    Y = np.zeros_like(data, dtype=np.float32)
    shift_coordinates(data, yup, xup, mshy, mshx, Y)
    return Y


def shift_frame(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """
    Returns frame, shifted by dy and dx

    Parameters
    ----------
    frame: Ly x Lx
    dy: int
        vertical shift amount
    dx: int
        horizontal shift amount

    Returns
    -------
    frame_shifted: Ly x Lx
        The shifted frame

    """
    # return shift(frame, (-dy, -dx), order=0)
    # thanks to Santi for the fix
    rolled = np.roll(frame, (-dy, -dx), axis=(0,1))
    dy *= -1; dx *= -1
    if dx < 0:
        rolled[:, dx:] = 0
    elif dx > 0:
        rolled[:, :dx] = 0
    if dy < 0:
        rolled[dy:, :] = 0
    elif dy > 0:
        rolled[:dy, :] = 0
    return rolled
    # return np.roll(frame, (-dy, -dx), axis=(0, 1))


@lru_cache(maxsize=5)
def mat_upsample(lpad: int, subpixel: int = 10):
    """
    upsampling matrix using gaussian kernels

    Parameters
    ----------
    lpad: int
    subpixel: int

    Returns
    -------
    Kmat: np.ndarray
    nup: int
    """
    lar = np.arange(-lpad, lpad + 1)
    larUP = np.arange(-lpad, lpad + .001, 1. / subpixel)
    nup = larUP.shape[0]
    Kmat = np.linalg.inv(kernelD(lar, lar)) @ kernelD(lar, larUP)
    return Kmat, nup

def kernelD(xs: np.ndarray, ys: np.ndarray, sigL: float = 0.85) -> np.ndarray:
    """
    Gaussian kernel from xs (1D array) to ys (1D array), with the "sigL" smoothing width for up-sampling kernels, (best between 0.5 and 1.0)

    Parameters
    ----------
    xs:
    ys
    sigL

    Returns
    -------

    """
    xs0, xs1 = np.meshgrid(xs, xs)
    ys0, ys1 = np.meshgrid(ys, ys)
    dxs = xs0.reshape(-1, 1) - ys0.reshape(1, -1)
    dys = xs1.reshape(-1, 1) - ys1.reshape(1, -1)
    K = np.exp(-(dxs**2 + dys**2) / (2 * sigL**2))
    return K

def nr_phasecorr(data: np.ndarray, maskMul, maskOffset, cfRefImg, snr_thresh, NRsm, xblock, yblock, maxregshiftNR, subpixel: int = 10, lpad: int = 3,):
    """
    Compute phase correlations for each block
    
    Parameters
    ----------
    data : nimg x Ly x Lx
    maskMul: ndarray
        gaussian filter
    maskOffset: ndarray
        mask offset
    cfRefImg
        FFT of reference image
    snr_thresh : float
        signal to noise ratio threshold
    NRsm
    xblock: float array
    yblock: float array
    maxregshiftNR: int
    subpixel: int
    lpad: int
        upsample from a square +/- lpad

    Returns
    -------
    ymax1
    xmax1
    cmax1
    """
    
    Kmat, nup = mat_upsample(lpad=3)

    nimg = data.shape[0]
    ly, lx = cfRefImg.shape[-2:]

    # maximum registration shift allowed
    lcorr = int(np.minimum(np.round(maxregshiftNR), np.floor(np.minimum(ly, lx) / 2.) - lpad))
    nb = len(yblock)

    # shifts and corrmax
    Y = np.zeros((nimg, nb, ly, lx), 'int16')
    for n in range(nb):
        yind, xind = yblock[n], xblock[n]
        Y[:,n] = data[:, yind[0]:yind[-1], xind[0]:xind[-1]]
    Y = addmultiply(Y, maskMul, maskOffset)

    Y = convolve(mov=Y, img=cfRefImg)

    # calculate ccsm
    lhalf = lcorr + lpad
    cc0 = np.real(
        np.block(
            [[Y[:, :, -lhalf:,    -lhalf:], Y[:, :, -lhalf:,    :lhalf + 1]],
             [Y[:, :, :lhalf + 1, -lhalf:], Y[:, :, :lhalf + 1, :lhalf + 1]]]
        )
    )
    cc0 = cc0.transpose(1, 0, 2, 3)
    cc0 = cc0.reshape(cc0.shape[0], -1)

    cc2 = [cc0, NRsm @ cc0, NRsm @ NRsm @ cc0]
    cc2 = [c2.reshape(nb, nimg, 2 * lcorr + 2 * lpad + 1, 2 * lcorr + 2 * lpad + 1) for c2 in cc2]
    ccsm = cc2[0]
    for n in range(nb):
        snr = np.ones(nimg, 'float32')
        for j, c2 in enumerate(cc2):
            ism = snr < snr_thresh
            if np.sum(ism) == 0:
                break
            cc = c2[n, ism, :, :]
            if j > 0:
                ccsm[n, ism, :, :] = cc
            snr[ism] = getSNR(cc, lcorr, lpad)

    # calculate ymax1, xmax1, cmax1
    mdpt = nup // 2
    ymax1 = np.empty((nimg, nb), np.float32)
    cmax1 = np.empty((nimg, nb), np.float32)
    xmax1 = np.empty((nimg, nb), np.float32)
    ymax = np.empty((nb,), np.int32)
    xmax = np.empty((nb,), np.int32)
    for t in range(nimg):
        ccmat = np.empty((nb, 2*lpad+1, 2*lpad+1), np.float32)
        for n in range(nb):
            ix = np.argmax(ccsm[n, t][lpad:-lpad, lpad:-lpad], axis=None)
            ym, xm = np.unravel_index(ix, (2 * lcorr + 1, 2 * lcorr + 1))
            ccmat[n] = ccsm[n, t][ym:ym + 2 * lpad + 1, xm:xm + 2 * lpad + 1]
            ymax[n], xmax[n] = ym - lcorr, xm - lcorr
        ccb = ccmat.reshape(nb, -1) @ Kmat
        cmax1[t] = np.amax(ccb, axis=1)
        ymax1[t], xmax1[t] = np.unravel_index(np.argmax(ccb, axis=1), (nup, nup))
        ymax1[t] = (ymax1[t] - mdpt) / subpixel + ymax
        xmax1[t] = (xmax1[t] - mdpt) / subpixel + xmax

    return ymax1, xmax1, cmax1


def getSNR(cc: np.ndarray, lcorr: int, lpad: int) -> float:
    """
    Compute SNR of phase-correlation.

    Parameters
    ----------
    cc: nimg x Ly x Lx
        The frame data to analyze
    lcorr: int
    lpad: int
        border padding width

    Returns
    -------
    snr: float
    """
    cc0 = cc[:, lpad:-lpad, lpad:-lpad].reshape(cc.shape[0], -1)
    # set to 0 all pts +-lpad from ymax,xmax
    cc1 = cc.copy()
    for c1, ymax, xmax in zip(cc1, *np.unravel_index(np.argmax(cc0, axis=1), (2 * lcorr + 1, 2 * lcorr + 1))):
        c1[ymax:ymax + 2 * lpad, xmax:xmax + 2 * lpad] = 0

    snr = np.amax(cc0, axis=1) / np.maximum(1e-10, np.amax(cc1.reshape(cc.shape[0], -1), axis=1))  # ensure positivity for outlier cases
    return snr