import cupy as cp
import numpy as n
from cupyx.scipy import fft as cufft
from cupyx.scipy import ndimage as cuimage
from scipy import ndimage
try:
    import mkl_fft
except:
    print("No MKL fft ")

def default_log(string, *args, **kwargs): 
    print(string)


def rigid_2d_reg_gpu(mov_cpu, mult_mask, add_mask, refs_f, max_reg_xy,
                    rmins, rmaxs, crosstalk_coeff = None, shift=True):
    
    nz, nt, ny, nx = mov_cpu.shape

    mov_gpu = cp.asarray(mov_cpu, dtype=cp.complex64)
    mult_mask_gpu = cp.asarray(mult_mask)
    add_mask_gpu = cp.asarray(add_mask)
    refs_f_gpu = cp.asarray(refs_f)
    ymaxs = cp.zeros((nz, nt))
    xmaxs = cp.zeros((nz, nt))
    
    ncc = max_reg_xy * 2 + 1
    phase_corr = cp.zeros((nt, ncc, ncc))

    if shift:
        mov_shifted = cp.zeros((nt,nz,ny,nx), dtype=cp.float32)
        mov_shifted[:] = mov_gpu.real.swapaxes(0,1)
    if crosstalk_coeff is not None: 
        mov_gpu = crosstalk_subtract(mov_gpu, crosstalk_coeff)

    for zidx in range(nz):
        mov_gpu[zidx] = clip_and_mask_mov(mov_gpu[zidx], rmins[zidx], rmaxs[zidx],
                          mult_mask_gpu[zidx], add_mask_gpu[zidx])
        mov_gpu[zidx] = convolve_2d_gpu(mov_gpu[zidx], refs_f_gpu[zidx])
        unwrap_fft_2d(mov_gpu[zidx].real, max_reg_xy, out=phase_corr)
        ymaxs[zidx], xmaxs[zidx] = get_max_cc_coord(phase_corr, max_reg_xy)
        if shift:
            for frame_idx in range(nt):
                mov_shifted[frame_idx, zidx] = cuimage.shift(mov_shifted[frame_idx, zidx],
                                shift=(-ymaxs[zidx, frame_idx], -xmaxs[zidx, frame_idx]),
                                output=mov_shifted[frame_idx, zidx])
    if shift:
        return mov_shifted, ymaxs, xmaxs
    return ymaxs, xmaxs



def rigid_2d_reg_cpu(mov_cpu, mult_mask, add_mask, refs_f, max_reg_xy,
                    rmins, rmaxs, crosstalk_coeff = None, shift=True):
    nz, nt, ny, nx = mov_cpu.shape
    mov = n.asarray(mov_cpu, dtype=n.complex64)
    ymaxs = n.zeros((nz, nt))
    xmaxs = n.zeros((nz, nt))
    ncc = max_reg_xy * 2 + 1
    phase_corr = n.zeros((nt, ncc, ncc))

    if shift:
        mov_shifted = n.zeros((nt,nz,ny,nx), dtype=n.float32)
        mov_shifted[:] = mov.real.swapaxes(0,1)
    if crosstalk_coeff is not None: 
        mov = crosstalk_subtract(mov, crosstalk_coeff)
    for zidx in range(nz):
        mov[zidx] = clip_and_mask_mov(mov[zidx], rmins[zidx], rmaxs[zidx],
                          mult_mask[zidx], add_mask[zidx], cp=n)
        mov[zidx] = convolve_2d_cpu(mov[zidx], refs_f[zidx])
        unwrap_fft_2d(mov[zidx].real, max_reg_xy, out=phase_corr, cp=n)
        ymaxs[zidx], xmaxs[zidx] = get_max_cc_coord(phase_corr, max_reg_xy, cp=n)
        if shift:
            for frame_idx in range(nt):
                mov_shifted[frame_idx, zidx] = ndimage.shift(mov_shifted[frame_idx, zidx],
                                shift=(-ymaxs[zidx, frame_idx], -xmaxs[zidx, frame_idx]),
                                output=mov_shifted[frame_idx, zidx])
    if shift:
        return mov_shifted, ymaxs, xmaxs
    return ymaxs, xmaxs


def get_max_cc_coord(phase_corr, max_reg_xy, cp=cp):
    nt, ncc, __ = phase_corr.shape
    phase_corr_flat = phase_corr.reshape(nt, ncc**2)
    argmaxs = cp.argmax(phase_corr_flat, axis=1)
    ymax = (argmaxs // ncc) - max_reg_xy
    xmax = (argmaxs %  ncc) - max_reg_xy
    return ymax, xmax

def clip_and_mask_mov(mov, rmin, rmax, mult_mask, add_mask, cp=cp):
    mov.real = cp.clip(mov.real, rmin, rmax, out=mov.real)
    mov *= mult_mask
    mov += add_mask
    return mov 

def unwrap_fft_2d(mov_float, nr, out=None, cp=cp):
    nt, ny, nx = mov_float.shape
    ncc = nr * 2 + 1
    if out is None:
        out = cp.zeros((nt, ncc, ncc), dtype=n.float32)
    out[:, :nr, :nr] = mov_float[:, -nr:, -nr:]
    out[:, nr:, :nr] = mov_float[:, :nr+1, -nr:]
    out[:, :nr, nr:] = mov_float[:, -nr:, :nr+1]
    out[:, nr:, nr:] = mov_float[:, :nr+1, :nr+1]
    return out 


def convolve_2d_gpu(mov, ref_f):
    mov = cufft.fft2(mov, axes=(1,2), overwrite_x=True)
    mov /= cp.abs(mov) + cp.complex64(1e-5)
    mov *= ref_f
    mov = cufft.ifft2(mov, axes=(1,2), overwrite_x=True)
    return mov

def convolve_2d_cpu(mov, ref_f):
    mov = mkl_fft.fft2(mov, axes=(1,2), overwrite_x=True)
    mov /= n.abs(mov) + n.complex64(1e-5)
    mov *= ref_f
    mov = mkl_fft.ifft2(mov, axes=(1,2), overwrite_x=True)
    return mov


def crosstalk_subtract(mov, crosstalk_coeff):
    nz, nt, ny, nx = mov.shape
    if nz <= 15: 
        return mov
    for i in range(nz - 15):
        mov[i + 15] -= crosstalk_coeff * mov[i]
    return mov
