import numpy as n
from functools import lru_cache
from . import utils
import time

try:
    from mkl_fft import fft2, ifft2
    using_scipy_fft = False
except ImportError:
    from scipy.fft import fft2, ifft2
    using_scipy_fft = True

try:
    import cupy as cp
    from cupyx.scipy import fft as cufft
    from cupyx.scipy import ndimage as cuimage
except ImportError:
    import numpy as cp
    from scipy.fft import fft as cufft
    from scipy import ndimage as cuimage

from .utils import default_log

def log_gpu_memory(mempool=None):
    if mempool is None:
        mempool = cp.get_default_memory_pool()
    n_blocks = mempool.n_free_blocks()
    total_pool_gb = mempool.total_bytes() / 1024**3
    used_pool_gb = mempool.used_bytes() / 1024**3
    #TODO confirm if this is correct
    string = "GPU RAM: %d blocks allocated, %2.2f / %2.2f GB used" \
             % (n_blocks, used_pool_gb, total_pool_gb)
    return string

def nonrigid_2d_reg_gpu(mov_gpu, mult_mask, add_mask, refs_nr_f, yblocks, xblocks, snr_thresh, smooth_mat, 
                        rmins=None, rmaxs=None,
                        max_shift=10, npad=3, n_smooth_iters=2, subpixel=5,
                        n_gpu_threads_per_block=512, log_cb = default_log):
    nr = max_shift + npad
    ncc = nr*2 + 1
    ncc_nopad = max_shift*2 + 1
    nt, nz, ny, nx = mov_gpu.shape
    __, nb, nby, nbx = refs_nr_f.shape

    mempool = cp.get_default_memory_pool()
    start_t = time.time()
    refs_nr_f = cp.asarray(refs_nr_f, cp.complex64)
    smooth_mat = cp.asarray(smooth_mat, cp.float32)
    mov_blocks = cp.zeros((nt,nz,nb,nby,nbx), cp.complex64)
    mult_mask = cp.asarray(mult_mask)
    add_mask = cp.asarray(add_mask)

    load_t = time.time()
    # log_cb("Allocated GPU array for non-rigid reg in %.2f sec" % ((load_t-start_t),), 4)
    # log_cb("Blocked movie is %2.2fGB" % (mov_blocks.nbytes/1024**3), 4)
    # log_cb(log_gpu_memory(mempool),4)
    # print(mov_gpu.std())
    if rmins is not None and rmaxs is not None:
        clip_t = time.time()
        for zidx in range(nz):
            mov_gpu[:,zidx] = clip_and_mask_mov(mov_gpu[:,zidx], rmins[zidx], rmaxs[zidx])
        # log_cb("Clipped movie in %.2f sec" % (time.time() - clip_t), 4)
    # print(mov_gpu.std())

    block_t = time.time()
    mov_blocks[:] = block_mov(mov_gpu, mov_blocks, yblocks, xblocks)
    mov_blocks *= mult_mask
    mov_blocks += add_mask
    
    # log_cb("Split movie into blocks in %.2f sec" % (time.time() - block_t),4)

    fft_t = time.time()
    mov_blocks[:] = convolve_2d_gpu(mov_blocks, refs_nr_f, axes=(3,4))
    phase_corr = cp.zeros((nt, nz, nb, ncc, ncc), dtype=cp.float32)
    phase_corr = unwrap_fft_2d(mov_blocks.real, nr = nr, out=phase_corr)
    # log_cb("Completed FFT of blocks and computed phase correlations in %.2f sec" %(time.time() - fft_t), 4)
    # print(mov_blocks.std())
    # print(phase_corr.std())
    smooth_t = time.time()
    pc, snrs = compute_snr_and_smooth(phase_corr, smooth_mat, n_smooth_iters,
                                      snr_thresh,log_cb=log_cb)
    # log_cb("Computed SNR and smoothed phase corrs in %.2f sec" % (time.time() - smooth_t),4)
    
    shift_t = time.time()
    ymaxs, xmaxs = get_subpixel_shifts(pc, max_shift, npad, subpixel, n_gpu_threads_per_block)
    # log_cb("Computed subpixel shifts in %.2f sec" % (time.time() - shift_t), 4)
    mempool.free_all_blocks()
    # log_cb(log_gpu_memory(mempool), 4)
    return ymaxs, xmaxs, snrs

def get_subpixel_shifts(pc, max_shift=10, npad=3, subpixel=5, n_thread_per_block=512):
    nt, nz, nb = pc.shape[:3]
    Kmat, nup = mat_upsample(lpad=npad, subpixel=subpixel)
    Kmat = cp.asarray(Kmat, cp.float32)
    mid = nup // 2
    pc_mat, ymaxs, xmaxs = crop_maxs(pc, npad, n_thread_per_block=n_thread_per_block)
    ymaxs -= max_shift; xmaxs -= max_shift
    pc_gaussian = pc_mat.reshape(nt,nz,nb,-1) @ Kmat

    ymaxs_sub, xmaxs_sub = cp.unravel_index(pc_gaussian.argmax(axis=-1), (nup,nup))

    # print(ymaxs.std())
    # print(ymaxs_sub.std())
    ymaxs = ymaxs.astype(cp.float32) + (ymaxs_sub.astype(cp.float32) - mid)/subpixel
    xmaxs = xmaxs.astype(cp.float32) + (xmaxs_sub.astype(cp.float32) - mid)/subpixel

    return ymaxs, xmaxs


def block_mov(mov_gpu, mov_blocks, yblocks, xblocks):
    nb = mov_blocks.shape[2]
    for bidx in range(nb):
        by0, by1 = yblocks[bidx]
        bx0, bx1 = xblocks[bidx]
        mov_blocks[:,:,bidx] = mov_gpu[:,:,by0:by1,bx0:bx1]
    return mov_blocks


def register_gpu():
    pass


def rigid_2d_reg_gpu(mov_cpu, mult_mask, add_mask, refs_f, max_reg_xy,
                    rmins, rmaxs, crosstalk_coeff = None, shift=True, 
                    min_pix_vals = None, fuse_and_pad=False, ypad=None, 
                    xpad=None, fuse_shift=None, new_xs=None, old_xs=None,
                    cavity_size = 15, log_cb=default_log):
    
    nz, nt, ny, nx = mov_cpu.shape
    start_t = time.time()
    mempool = cp.get_default_memory_pool()
    if not fuse_and_pad: mov_gpu = cp.asarray(mov_cpu, dtype=cp.complex64)
    if fuse_and_pad: mov_gpu = cp.asarray(mov_cpu, dtype=cp.float32)
    mult_mask_gpu = cp.asarray(mult_mask)
    add_mask_gpu = cp.asarray(add_mask)
    refs_f_gpu = cp.asarray(refs_f)
    ymaxs = cp.zeros((nz, nt), dtype=cp.float32)
    xmaxs = cp.zeros((nz, nt), dtype=cp.float32)
    cmaxs = cp.zeros((nz, nt), dtype=cp.float32)
    ncc = int(max_reg_xy * 2 + 1)
    phase_corr = cp.zeros((nt, ncc, ncc))

    load_t = time.time()
    # log_cb("Loaded mov and masks to GPU for rigid reg in %.2f sec" % ((load_t-start_t),), 4)

    if min_pix_vals is not None:
        # log_cb("Subtracting min pix vals to enforce positivity", 4)
        min_pix_vals = cp.asarray(min_pix_vals, dtype=mov_gpu.dtype)
        mov_gpu[:] -= min_pix_vals[:nz, cp.newaxis, cp.newaxis, cp.newaxis]

    if crosstalk_coeff is not None: 
        # log_cb("Subtracting crosstalk", 4)
        mov_gpu = utils.crosstalk_subtract(mov_gpu, crosstalk_coeff, cavity_size)
    
    if fuse_and_pad:
        # log_cb("Fusing and padding movie",4)
        mov_gpu = fuse_and_pad_gpu(mov_gpu, fuse_shift, ypad, xpad, new_xs, old_xs)
        nz,nt,ny,nx = mov_gpu.shape
        # log_cb("GPU Mov of shape %d, %d, %d, %d; %.2f GB" % (nz, nt, ny, nx, mov_gpu.nbytes/(1024**3)),4)
        mempool.free_all_blocks()
        # log_cb(log_gpu_memory(mempool), 4)

    if shift:
        # log_cb("Allocating memory for shifted movie", 4)
        mov_shifted = cp.zeros((nt,nz,ny,nx), dtype=cp.float32)
        mov_shifted[:] = mov_gpu.real.swapaxes(0,1).copy()

    # print("mov_shifted before reg, min: %.2f, max: %.2f" % (mov_shifted[:,10].min(), mov_shifted[:,10].max()))


    # log_cb(log_gpu_memory(mempool), 4)
    reg_t = 0; shift_t = 0
    for zidx in range(nz):
        reg_tic = time.time()
        # log_cb("Registering plane %d" % (zidx,), 4)
        mov_gpu[zidx] = clip_and_mask_mov(mov_gpu[zidx], rmins[zidx], rmaxs[zidx],
                          mult_mask_gpu[zidx], add_mask_gpu[zidx])
        mov_gpu[zidx] = convolve_2d_gpu(mov_gpu[zidx], refs_f_gpu[zidx])
        unwrap_fft_2d(mov_gpu[zidx].real, max_reg_xy, out=phase_corr)
        ymaxs[zidx], xmaxs[zidx], cmaxs[zidx] = get_max_cc_coord(phase_corr, max_reg_xy)
        reg_t += (time.time() - reg_tic)
        
        if shift:
            shift_tic = time.time()
            log_cb("Shifting plane %d" % (zidx,), 4)
            xmax_z, ymax_z = xmaxs[zidx].get(), ymaxs[zidx].get()
                
            # if zidx == 10:
            #     print("mov_shifted before shift, min: %.2f, max: %.2f" % (mov_shifted[:,10].min(), mov_shifted[:,10].max()))
            for frame_idx in range(nt):
                mov_shifted[frame_idx, zidx] = shift_frame(mov_shifted[frame_idx, zidx],
                                dy=ymax_z[frame_idx], dx=xmax_z[frame_idx])
            shift_t += (time.time() - shift_tic)

    
    # print("mov_shifted after shift, min: %.2f, max: %.2f" % (mov_shifted[:,10].min(), mov_shifted[:,10].max()))

    # log_cb("Registered batch in %.2f sec"  % reg_t, 3)
    # if shift: 
    #     log_cb("Shifted batch in %.2f sec" % shift_t, 3)
    # log_cb(log_gpu_memory(mempool), 4)
    mempool.free_all_blocks()
    # log_cb("Freeing all blocks", 3)
    # log_cb(log_gpu_memory(mempool), 4)
    if shift:
        return mov_shifted, ymaxs, xmaxs, cmaxs
    return ymaxs, xmaxs, cmaxs

def rigid_2d_reg_cpu(mov_cpu, mult_mask, add_mask, refs_f, max_reg_xy,
                    rmins, rmaxs, crosstalk_coeff = None, shift=True, cavity_size = 15):
    nz, nt, ny, nx = mov_cpu.shape
    mov = n.asarray(mov_cpu, dtype=n.complex64)
    ymaxs = n.zeros((nz, nt), dtype=n.int16)
    xmaxs = n.zeros((nz, nt), dtype=n.int16)
    cmaxs = n.zeros((nz, nt), dtype=n.float32) 
    ncc = max_reg_xy * 2 + 1
    phase_corr = n.zeros((nt, ncc, ncc))

    if shift:
        mov_shifted = n.zeros((nt,nz,ny,nx), dtype=n.float32)
        mov_shifted[:] = mov.real.swapaxes(0,1)
    if crosstalk_coeff is not None: 
        mov = utils.crosstalk_subtract(mov, crosstalk_coeff, cavity_size)
    for zidx in range(nz):
        mov[zidx] = clip_and_mask_mov(mov[zidx], rmins[zidx], rmaxs[zidx],
                          mult_mask[zidx], add_mask[zidx], cp=n)
        mov[zidx] = convolve_2d_cpu(mov[zidx], refs_f[zidx])
        unwrap_fft_2d(mov[zidx].real, max_reg_xy, out=phase_corr, cp=n)
        ymaxs[zidx], xmaxs[zidx], cmaxs[zidx] = get_max_cc_coord(phase_corr, max_reg_xy, cp=n)
        if shift:
            for frame_idx in range(nt):
                mov_shifted[frame_idx, zidx] = shift_frame(mov_shifted[frame_idx, zidx],
                                dy=ymaxs[zidx, frame_idx], dx=xmaxs[zidx, frame_idx], cp=n)
                
    
    if shift:
        return mov_shifted, ymaxs, xmaxs
    return ymaxs, xmaxs

def get_max_cc_coord_old(phase_corr, max_reg_xy, cp=cp):
    nt, ncc, __ = phase_corr.shape
    phase_corr_flat = phase_corr.reshape(nt, ncc**2)
    argmaxs = cp.argmax(phase_corr_flat, axis=1)
    ymax = (argmaxs // ncc) - max_reg_xy
    xmax = (argmaxs %  ncc) - max_reg_xy
    return ymax, xmax

def get_max_cc_coord(phase_corr, max_reg_xy, cp=cp):
    """
    This function finds where the highest correlation to find the value of the coreelation 
    and the x/y shifts needed to maximise correlation 

    Parameters
    ----------
    phase_corr : ndarray (nT, ncc, ncc)
        The phase correlation image for each frame
    max_reg_xy : int
        The maximum size shift the function allows
    cp : function class, optional
        Does the function use GPU (cp) or CPU (n), by default cp

    Returns
    -------
    ndarray x3
        returns the values of ymax, xmax and cmax for each frame
    """
    nt, ncc, __ = phase_corr.shape
    phase_corr_flat = phase_corr.reshape(nt, ncc**2)
    # get locations of the maximum phase correlation
    argmaxs = cp.argmax(phase_corr_flat, axis=1)
    # get the value of the maximum phase correlation
    cmax = cp.max(phase_corr_flat, axis = 1)
    ymax = (argmaxs // ncc) - max_reg_xy
    xmax = (argmaxs %  ncc) - max_reg_xy
    return ymax, xmax, cmax

def clip_and_mask_mov(mov, rmin, rmax, mult_mask=None, add_mask=None, cp=cp):
    if rmin is not None and rmax is not None: mov.real = cp.clip(mov.real, rmin, rmax, out=mov.real)
    if mult_mask is not None: mov *= mult_mask
    if add_mask is not None:  mov += add_mask
    return mov 

def unwrap_fft_2d(mov_float, nr, out=None, cp=cp):
    nt = mov_float.shape[0]
    ny,nx = mov_float.shape[-2:]
    ndim = len(mov_float.shape)
    ncc = nr * 2 + 1
    if out is None:
        out = cp.zeros((nt, ncc, ncc), dtype=n.float32)
    ndim_out = len(out.shape)
    idxs_out = [slice(None) for i in range(ndim_out)]
    idxs_mov = [slice(None) for i in range(ndim)]

    idxs_out[-2] = slice(0, nr);   idxs_out[-1] = slice(0,nr)
    idxs_mov[-2] = slice(-nr, ny); idxs_mov[-1] = slice(-nr, nx)
    # print(nr, ny)
    # print(idxs_out)
    # print(idxs_mov)
    # print(mov_float.shape)
    # print(out[tuple(idxs_out)].shape)
    # print(mov_float[tuple(idxs_mov)].shape)
    out[tuple(idxs_out)] = mov_float[tuple(idxs_mov)]
    idxs_out[-2] = slice(nr,ncc);   idxs_out[-1] = slice(0,nr)
    idxs_mov[-2] = slice(0,nr+1); idxs_mov[-1] = slice(-nr, nx)
    out[tuple(idxs_out)] = mov_float[tuple(idxs_mov)]
    idxs_out[-2] = slice(0, nr);   idxs_out[-1] = slice(nr,ncc)
    idxs_mov[-2] = slice(-nr, ny); idxs_mov[-1] = slice(0, nr+1)
    out[tuple(idxs_out)] = mov_float[tuple(idxs_mov)]
    idxs_out[-2] = slice(nr, ncc);   idxs_out[-1] = slice(nr,ncc)
    idxs_mov[-2] = slice(0, nr+1);  idxs_mov[-1] = slice(0, nr+1)
    out[tuple(idxs_out)] = mov_float[tuple(idxs_mov)]

    # out[:, :nr, :nr] = mov_float[:, -nr:, -nr:]
    # out[:, nr:, :nr] = mov_float[:, :nr+1, -nr:]
    # out[:, :nr, nr:] = mov_float[:, -nr:, :nr+1]
    # out[:, nr:, nr:] = mov_float[:, :nr+1, :nr+1]
    return out 

# def make_ref_z_1d(ref_real_cpu)

def unwrap_fft_1d(mov_float, nr):
    out = cp.concatenate([mov_float[:, -nr:],mov_float[:, :nr+1]])
    return out 

def convolve_1d_gpu(mov, ref_f, phasenorm=False):
    mov[:] = cufft.fft(mov, axis=1, overwrite_x=True)
    if phasenorm: 
        mov /= cp.abs(mov) + cp.complex64(1e-5)
    mov *= ref_f
    mov[:] = cufft.ifft(mov, axis=1, overwrite_x=True)
    return mov

def convolve_2d_gpu(mov, ref_f, axes=(1,2)):
    mov[:] = cufft.fft2(mov, axes=axes, overwrite_x=True)
    for i in range(mov.shape[0]): # loop to reduce memory usage
        mov[i] /= cp.abs(mov[i]) + cp.complex64(1e-5)
    mov *= ref_f
    mov[:] = cufft.ifft2(mov, axes=axes, overwrite_x=True)
    return mov

def convolve_2d_cpu(mov, ref_f):
    if using_scipy_fft:
        mov[:] = fft2(mov, axes=(1,2), overwrite_x=True)
    else:
        mov[:] = fft2(mov, axes=(1,2))
    mov /= n.abs(mov) + n.complex64(1e-5)
    mov *= ref_f
    if using_scipy_fft:
        mov[:] = ifft2(mov, axes=(1,2), overwrite_x=True)
    else:
        mov[:] = ifft2(mov, axes=(1,2))
    return mov

#TODO xpad/ypad should be integer ?
def fuse_and_pad_gpu(mov_gpu, fuse_shift, ypad, xpad, new_xs, old_xs):
    nz, nt, ny, nx = mov_gpu.shape
    n_stitches = len(new_xs) - 1
    n_xpix_lost_fusing = n_stitches * fuse_shift
    nyn = ny + ypad.sum()
    nxn = nx + xpad.sum() - n_xpix_lost_fusing

    mov_pad = cp.zeros((nz, nt, nyn, nxn), dtype=cp.complex64)
    for strip_idx in range(len(new_xs)):
        nx0,nx1 = new_xs[strip_idx]
        ox0,ox1 = old_xs[strip_idx]
        mov_pad[:,:,:ny, nx0:nx1] = mov_gpu[:,:,:,ox0:ox1]

    return mov_pad

def fuse_and_pad(mov, fuse_shift, ypad, xpad, new_xs, old_xs):
    nz, nt, ny, nx = mov.shape
    n_stitches = len(new_xs) - 1
    n_xpix_lost_fusing = n_stitches * fuse_shift
    nyn = ny + ypad.sum()
    nxn = nx + xpad.sum() - n_xpix_lost_fusing

    mov_pad = n.zeros((nz, nt, nyn, nxn), dtype=mov.dtype)
    for strip_idx in range(len(new_xs)):
        nx0,nx1 = new_xs[strip_idx]
        ox0,ox1 = old_xs[strip_idx]
        mov_pad[:,:,:ny, nx0:nx1] = mov[:,:,:,ox0:ox1]

    return mov_pad

def shift_frame(frame, dy, dx, cp = cp):

    # return shift(frame, (-dy, -dx), order=0)
    # thanks to Santi for the fix
    frame[:] = cp.roll(frame, (-dy, -dx), axis=(0,1))
    dy *= -1; dx *= -1
    if dx < 0:
        frame[:, dx:] = 0
    elif dx > 0:
        frame[:, :dx] = 0
    if dy < 0:
        frame[dy:, :] = 0
    elif dy > 0:
        frame[:dy, :] = 0
    return frame

# from suite2p
def kernelD(xs: n.ndarray, ys: n.ndarray, sigL: float = 0.85) -> n.ndarray:
    """
    Gaussian kernel from xs (1D array) to ys (1D array), with the 'sigL' smoothing width for up-sampling kernels, (best between 0.5 and 1.0)

    Parameters
    ----------
    xs:
    ys
    sigL

    Returns
    -------

    """
    xs0, xs1 = n.meshgrid(xs, xs)
    ys0, ys1 = n.meshgrid(ys, ys)
    dxs = xs0.reshape(-1, 1) - ys0.reshape(1, -1)
    dys = xs1.reshape(-1, 1) - ys1.reshape(1, -1)
    K = n.exp(-(dxs ** 2 + dys ** 2) / (2 * sigL ** 2))
    return K

# from suite2p 
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
    lar = n.arange(-lpad, lpad + 1)
    larUP = n.arange(-lpad, lpad + .001, 1. / subpixel)
    nup = larUP.shape[0]
    Kmat = n.linalg.inv(kernelD(lar, lar)) @ kernelD(lar, larUP)
    return Kmat, nup

def compute_snr_and_smooth(phase_corr, smooth_mat, n_smooth_iters = 2, 
                           snr_thresh=1.2, log_cb=default_log):
    pc = phase_corr.copy()
    pc_smooth = pc.copy()
    for i in range(n_smooth_iters):
        snrs = get_snr(pc)
        idx_to_smooth = (snrs < snr_thresh)
        n_low_snr = idx_to_smooth.sum()
        # log_cb("Iter %d: %d/%d blocks below SNR thresh" % (i, n_low_snr, snrs.size), 4)
        if n_low_snr < 1: break
        pc_smooth = cp.moveaxis(cp.tensordot(smooth_mat, pc_smooth, ((1,), (2,))),0,2)
        pc[idx_to_smooth] = pc_smooth[idx_to_smooth]
    snrs = get_snr(pc)
    n_low_snr = (snrs < snr_thresh).sum()
    # log_cb("Iter %d: %d/%d blocks below SNR thresh" % (i, n_low_snr, snrs.size), 4)
    return pc, snrs



def get_snr(phase_corr, npad=3, kernel=None, n_thread_per_block=512, slow=False):
    phase_corr = phase_corr.copy()
    nt, nz, nb, nccy, nccx = phase_corr.shape
    nball = nt*nz*nb
    assert nccy == nccx; ncc = nccy

    if kernel is None:
        kernel = get_kernel_zero_around_max()

    max_nopad = phase_corr[:,:,:,npad:-npad, npad:-npad].max(axis=(-1,-2))
    phase_corr = phase_corr.reshape(nt,nz,nb, ncc**2)
    argmaxs = cp.argmax(phase_corr, axis=-1)
    argmax_ys, argmax_xs = cp.unravel_index(argmaxs, (ncc,ncc))
    phase_corr = phase_corr.reshape(nt,nz,nb, ncc, ncc)

    xmin = (argmax_xs - npad); xmin[xmin < 0] = 0
    xmax = (argmax_xs + npad); xmax[xmax > ncc] = ncc
    ymin = (argmax_ys - npad); ymin[ymin < 0] = 0
    ymax = (argmax_ys + npad); ymax[ymax > ncc] = ncc

    # fill the area around the maximum argument of phase_corr with 0s
    if slow:
        for tx in range(nt):
            for zx in range(nz):
                for bx in range(nb):
                    phase_corr[tx, zx, bx, ymin[tx,zx,bx]:ymax[tx,zx,bx], xmin[tx,zx,bx]:xmax[tx,zx,bx]] = 0
    else:
        n_blocks = int(n.ceil(nball / n_thread_per_block))
        kernel((n_blocks,), (n_thread_per_block,),
            (xmin, xmax, ymin, ymax, phase_corr, cp.uint32(ncc), cp.uint32(nball)))
        
    max_zerod = phase_corr.max(axis=(-1, -2))


    snrs = max_nopad / cp.maximum(1e-10, max_zerod)

    return snrs

def crop_maxs(pc, npad, kernel=None, n_thread_per_block=512):
    nt,nz,nb,ncc_pad,__ = pc.shape
    nball = nb*nz*nt
    ncc_nopad = ncc_pad - npad * 2

    pc_nopad = cp.zeros((nt, nz, nb, ncc_nopad, ncc_nopad), dtype=cp.float32)
    pc_nopad[:] =  pc[:,:,:,npad:-npad, npad:-npad]

    argmaxs = cp.argmax(pc_nopad, axis=(-1,-2))

    ymaxs, xmaxs = cp.unravel_index(argmaxs, (ncc_nopad, ncc_nopad))

    xmin = (xmaxs - npad); xmin[xmin < 0] = 0
    xmax = (xmaxs + npad+1); xmax[xmax > ncc_nopad] = ncc_nopad
    ymin = (ymaxs - npad); ymin[ymin < 0] = 0
    ymax = (ymaxs + npad+1); ymax[ymax > ncc_nopad] = ncc_nopad

    if kernel is None:
        kernel = get_kernel_crop_around_max()

    npadmat = npad * 2 + 1

    pc_mat = cp.zeros((nt, nz, nb, npadmat, npadmat), cp.float32)
    n_blocks = int(n.ceil(nball / n_thread_per_block))
    kernel((n_blocks,), (n_thread_per_block,), (xmin, xmax, ymin,ymax,
                                pc_nopad, pc_mat, cp.uint32(npadmat), cp.uint32(ncc_nopad), cp.uint32(nball)))

    return pc_mat, ymaxs, xmaxs

def get_kernel_crop_around_max():
    kernel_crop_around_max = cp.RawKernel(r'''
    extern "C" __global__
    void crop_around_max(long long* xmin, long long* xmax, long long* ymin, long long* ymax, 
                        float* in, float* out, unsigned int npadmat, unsigned int ncc, unsigned int max){
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int xx; int yy;
        int i; int j;
        if (tid < max){
            i = 0;
            for (yy = ymin[tid]; yy < ymax[tid]; yy++){
                j = 0;
                for (xx = xmin[tid]; xx < xmax[tid]; xx++) {
                    out[(tid * npadmat * npadmat) + i*npadmat + j] = in[(tid * ncc * ncc) + (yy * ncc) + xx];
                    j++;
                }
                i++;
            }
        }
    }
    ''', "crop_around_max"
    )
    return kernel_crop_around_max

def get_kernel_zero_around_max():
    # https://docs.cupy.dev/en/stable/user_guide/kernel.html#kernel-arguments
    kernel_zero_around_max = cp.RawKernel(r'''
    extern "C" __global__
    void zero_around_max(long long* xmin, long long* xmax, long long* ymin, long long* ymax, float* out, 
                        unsigned int ncc, unsigned int max){
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int xx; int yy;
        if (tid < max){
            for (yy = ymin[tid]; yy < ymax[tid]; yy++){
                for (xx = xmin[tid]; xx < xmax[tid]; xx++) {
                    out[(tid * ncc * ncc) + (yy * ncc) + xx] = 0;
                }
            }
        }
    }
    ''', "zero_around_max"
    )
    return kernel_zero_around_max