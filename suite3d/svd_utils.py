import os
import dask
from dask import array as darr
from dask_image.ndfilters import uniform_filter as dask_uniform_filter
import numpy as n
import zarr
import time
from scipy import ndimage

def default_log(string, *args, **kwargs): 
    print(string)


def block_and_svd(mov_reg, n_comp, block_shape= (1, 128, 128), block_overlaps=(0, 18, 18),
                  t_chunk=4000, pix_chunk= None,t_save_chunk=100, comp_chunk=100, n_svd_blocks_per_batch = 4, svd_dir=None,
                  block_validity=None, log_cb=default_log, flip_shape=False, end_batch=None,
                  start_batch = 0, run_svd=True, other_info = {}):
    if not flip_shape:
        nz, nt, ny, nx = mov_reg.shape
    elif flip_shape:
        nt, nz, ny, nx = mov_reg.shape
    blocks, grid_shape = make_blocks((nz, ny, nx), block_shape, block_overlaps)
    if pix_chunk is None:
        pix_chunk = n.product(block_shape)
    n_blocks = blocks.shape[1]

    log_cb("Will compute SVD in %d blocks in a grid shaped %s" %
           (n_blocks, str(grid_shape)), 1)

    n_batches = n.ceil(n_blocks / n_svd_blocks_per_batch)
    log_cb("Batching %d blocks together, for a total of %d batches. Pix chunksize: %d" %
           (n_svd_blocks_per_batch, n_batches, pix_chunk))

    svd_block_dir = os.path.join(svd_dir, 'blocks')
    os.makedirs(svd_block_dir, exist_ok=True)
    log_cb("Saving SVD blocks in %s" % svd_block_dir)
    svd_info_path = os.path.join(svd_dir, 'svd_info.npy')
    # return
    if start_batch == 0:
        svd_info = {
            'n_blocks': n_blocks,
            'block_shape': block_shape,
            'block_overlaps': block_overlaps,
            'blocks': blocks,
            'grid_shape': grid_shape,
            'mov_shape': mov_reg.shape,
            'n_comps': n_comp,
            'svd_dirs': []
        }
        svd_info.update(other_info)
    else:
        print("Loading SVD info from %s" % svd_info_path)
        svd_info = n.load(svd_info_path, allow_pickle=True).item()
    if not run_svd: return svd_info
    batch_idx = start_batch + 1
    print("Starting with batch %d" % start_batch)
    for batch_start in range(start_batch*n_svd_blocks_per_batch, n_blocks, n_svd_blocks_per_batch):
        tic = time.time()
        batch_end = min(batch_start + n_svd_blocks_per_batch, n_blocks)
        log_cb("Starting batch %d / %d, blocks %d - %d" %
               (batch_idx, n_batches, batch_start, batch_end), 2)
        to_compute = []
        for block_idx in range(batch_start, batch_end):
            if block_validity is not None:
                if not block_validity[block_idx]:
                    continue
            slices = [slice(x[0], x[1], 1) for x in blocks[:, block_idx]]
            if not flip_shape:
                block = mov_reg[slices[0], :, slices[1],
                                slices[2]].swapaxes(0, 1).reshape(nt, -1)
            else:
                block = mov_reg[:, slices[0], slices[1],
                                slices[2]].reshape(nt, -1)

            zarr_dir = os.path.join(svd_block_dir, '%04d' % block_idx)
            os.makedirs(zarr_dir, exist_ok=True)
            temp = run_svd_on_block(block.rechunk((t_chunk, pix_chunk)),
                                    n_comp, save_zarr=True, svd_dir=zarr_dir,
                                    timechunk=t_save_chunk, compchunk=comp_chunk)
            svd_info['svd_dirs'].append(zarr_dir)
            to_compute.append(temp)
        log_cb("Sending batch %d to dask" % batch_idx, 2)
        dask_tic = time.time()
        xx = darr.compute(to_compute)
        dask_toc = time.time() - dask_tic
        log_cb("Dask completed in %.3f sec" % dask_toc, 2)

        log_cb("Saving svd_info to %s" % svd_info_path, 2)
        n.save(svd_info_path, svd_info)

        full_toc = time.time() - tic
        log_cb("Batch %d completed in %.3f" % (batch_idx, full_toc))
        if batch_idx == 1:
            rolling_mean_batch_time = full_toc
        else:
            alpha = 0.8
            rolling_mean_batch_time = full_toc * \
                (1-alpha) + rolling_mean_batch_time * alpha
        est_remaining_time = ((n_batches - batch_idx)
                              * rolling_mean_batch_time)
        est_time_str = time.strftime(
            "%Hh%Mm%Ss", time.gmtime(est_remaining_time))
        log_cb("Estimated time remaining for %d batches: %s" %
               (n_batches - batch_idx, est_time_str))

        if end_batch is not None and end_batch == batch_idx:
            print("Breaking")
            return svd_info
        batch_idx += 1
    return svd_info

import time

def reconstruct(svd_info, t_indices):
    pass

def get_overlap_mask(block_shape, overlaps, bsize=3, fsize=1.4):
    mask = n.zeros(block_shape)
    noz, noy, nox = overlaps
    nbz, nby, nbx = block_shape
    mask[(noz//bsize):(nbz-noz//bsize),(noy//bsize) : (nby-noy//bsize),(nox//bsize) : (nbx-nox//bsize)] = 1
    mask = ndimage.uniform_filter(mask, (noz//fsize, noy//fsize, nox//fsize))
    return mask

def reconstruct_overlapping_movie(svd_info, t_indices, filt_size = None, block_chunks=1, normalize=True, n_comps = None, crop_z = None, log_cb=default_log):
    
    # print("RECONSTRUCT START")
    tic = time.time()
    ntx = t_indices[1] - t_indices[0]
    t_batch_size = ntx
    all_blocks = reconstruct_movie(svd_info, t_batch_size = t_batch_size, return_blocks=True, block_chunks=block_chunks, n_comps = n_comps, t_indices=t_indices)
    
    nz, nt, ny, nx = svd_info['mov_shape']
    block_limits = svd_info['blocks']
    nbz, nby, nbx = svd_info['block_shape']
    n_blocks = svd_info['n_blocks']
    
    if crop_z is not None:
        nz = crop_z[1] - crop_z[0]
        valid_blocks = []
        for i in range(n_blocks):
            if block_limits[0,i,0] in range(*crop_z):
                valid_blocks.append(i)
        valid_blocks = n.array(valid_blocks)
        all_blocks = all_blocks[valid_blocks]
        n_blocks = all_blocks.shape[0]

    mov3d = n.zeros((ntx, nz, ny, nx), dtype=n.float32)
    norm3d = n.zeros((nz,ny,nx))
    noz, noy, nox = svd_info['block_overlaps']
    mask = n.zeros((nbz, nby, nbx)) 
    bsize = 3
    mask[(noz//bsize):(nbz-noz//bsize),(noy//bsize) : (nby-noy//bsize),(nox//bsize) : (nbx-nox//bsize)] = 1
    fsize = 1.4
    mask = ndimage.uniform_filter(mask, (noz//fsize, noy//fsize, nox//fsize))
    # print(all_blocks.shape)
    # all_blocks_batch = all_blocks[:, t_indices[0]:t_indices[1]]
    # print(all_blocks.shape)
    # return
    log_cb("Sending all blocks to dask to compute",3, log_mem_usage=True)
    all_blocks_batch = all_blocks.compute()
    log_cb("Dask reconstruction complete", 3, log_mem_usage=True)
    # print("RECONSTRUCT - DASK COMPLETE at t %.2f" % (time.time()-tic))
    for i in range(n_blocks):
        zz,yy,xx = block_limits[:,i]
        block = all_blocks_batch[i].reshape(
            t_batch_size, nbz, nby, nbx) * mask[n.newaxis]
        mov3d[:,zz[0]:zz[1], yy[0]:yy[1], xx[0]:xx[1]] += block
        norm3d[zz[0]:zz[1], yy[0]:yy[1], xx[0]:xx[1]] += mask

        
    if normalize:
        norm3d[norm3d < 1e-5] = n.inf
        # print("RECONSTRUCT - ALL COMPLETE at t %.2f" % (time.time()-tic))
        return mov3d / norm3d
    else: 
        # print("RECONSTRUCT - ALL COMPLETE at t %.2f" % (time.time()-tic))
        return mov3d, norm3d

def reconstruct_movie(svd_info, t_batch_size=None, return_blocks=False, block_chunks=1, n_comps=None, old_func=False, t_indices=None):
    if type(svd_info) == str:
        svd_info = n.load(os.path.join(svd_info, 'svd_info.npy'), allow_pickle=True).item()
    svd_dirs = svd_info['svd_dirs']

    if n_comps is None: n_comps = svd_info['n_comps']
    us,ss,vs = load_stack_usvs(svd_dirs, n_comps,stack_axis=0, t_batch_size=t_batch_size, t_indices=t_indices)
    if old_func: 
        print("probably will go crazy")
        blocks_dn = reconstruct_from_stack_old(us,ss,vs, time_chunks=t_batch_size)
    else: 
        blocks_dn = reconstruct_from_stack(us,ss,vs, n_comps, block_chunks, t_batch_size)
        if return_blocks: return blocks_dn
        blocks_dn = darr.swapaxes(blocks_dn, 0, 1)
    if return_blocks:
        return blocks_dn
    print(blocks_dn.shape)
    mov_dn = reshape_reconstructed_blocks(blocks_dn, svd_info['block_shape'], 
                                            svd_info['mov_shape'], svd_info['grid_shape'],
                                          rechunk_ts=t_batch_size)
    return mov_dn


def reconstruct(u, s, v, n_comp=None, dtype=n.float32, dask=True, reshape=None, rechunk_comps=None):
    if n_comp is not None:
        if type(n_comp) in (float, int):
            s = s[:n_comp]
            u = u[:, :n_comp]
            v = v[:n_comp]
        else:
            s = s[n_comp]
            u = u[:, n_comp]
            v = v[n_comp]
    if rechunk_comps is not None:
        s = s.rechunk(rechunk_comps)
        u = u.rechunk((None, rechunk_comps))
        v = v.rechunk((rechunk_comps, None))

    if dask:
        sdiag = darr.diag(s)
    else:
        sdiag = n.diag(s)
    us = (u @ sdiag)
    usv = us.astype(dtype) @ v.astype(dtype)
    if reshape is not None:
        usv = usv.reshape(reshape)
    return usv

def reconstruct_from_stack(us, ss, vs, n_comp=None, block_chunks=1, time_chunks=None, dtype=n.float32):
    if n_comp: n_comp = ss.shape[1]
    us = us.rechunk((block_chunks, time_chunks, n_comp))[:,:,:n_comp]
    ss = ss.rechunk((block_chunks, n_comp))[:,:n_comp]
    vs = vs.rechunk((block_chunks, n_comp, None))[:,:n_comp]
    vss = ss[:,:,n.newaxis] * vs
    usv = darr.matmul(us.astype(dtype), vss.astype(dtype))
    return usv

def reconstruct_from_stack_old(us, ss, vs, rechunk_comps='max', n_comp_reconstruct=None, dtype=None, stack_axis=1, 
                               time_chunks = None):
    n_blocks, n_comp = ss.shape
    if rechunk_comps == 'max':
        rechunk_comps = n_comp
    mov3d = []
    if time_chunks is not None: us = us.rechunk((time_chunks, None, None))
    for i in range(n_blocks):
        mov_i = reconstruct(us[i], ss[i], vs[i], rechunk_comps=rechunk_comps, n_comp=n_comp_reconstruct)
        if dtype is not None:
            mov_i = mov_i.astype(dtype)
        mov3d.append(mov_i)
    mov3d = darr.stack(mov3d, axis=stack_axis)
    return mov3d

def reshape_reconstructed_blocks(blocks, block_shape, mov_shape, grid_shape, rechunk_ts = None):
    nz, nt, ny, nx = mov_shape
    bz, by, bx = block_shape
    gz, gy, gx = grid_shape
    blocks = blocks.reshape(nt, gz, gy, gx, by,bx, limit='100GiB')
    blocks = darr.swapaxes(blocks, 3,4)
    blocks = blocks.reshape(nt, nz, ny, nx, limit='100GiB')
    if rechunk_ts is not None:
        blocks = blocks.rechunk((rechunk_ts, nz, ny, nx))
        
    return blocks

def run_svd_on_block(block, n_comp, svd_dir=None, save_zarr=True, timechunk=200, compchunk=100,
                     svd_compressed_params = {}, use_regular_svd=False):
    if use_regular_svd:
        u,s,v = darr.linalg.svd(block)
    else:
        u,s,v = darr.linalg.svd_compressed(block, k=n_comp, **svd_compressed_params)
    if not save_zarr:
        return u,s,v
    u = u.rechunk((timechunk,compchunk))
    s = s.rechunk((compchunk,))
    v = v.rechunk((compchunk, -1))
    zarrs = {}
    temp_vals = []
    for zarr_name, arr in zip(['u','s','v'], (u,s,v)):
        zarr_path = os.path.join(svd_dir, '%s.zarr' % zarr_name)
        zarrs[zarr_name] = zarr.open(zarr_path, compressor=None, mode='w',
                                        shape=arr.shape, chunks=arr.chunksize, dtype=arr.dtype)
        temp_vals.append(arr.store(zarrs[zarr_name], compute=False, lock=False))
        
    return temp_vals


def make_blocks_1d(axis_size, block_size, overlap, nonoverlapping_mask=False):
    # solve overlap <= block_size - (axis_size - block_size) / (n_blocks - 1)
    # to get:
    # n_blocks >= (ax - bl) / (bl - ov)  + 1
    n_blocks = int(n.ceil((axis_size - block_size) /
                   (block_size - overlap) + 1))
    block_starts = n.linspace(0, axis_size-block_size, n_blocks).astype(int)
    block_ends = block_starts + block_size
    blocks = n.stack([block_starts, block_ends], axis=1)

    if nonoverlapping_mask:
        for i in range(blocks.shape[0]-1):
            mean = int(n.floor((blocks[i,1] + blocks[i+1,0])/2))
            blocks[i,1] = mean
            blocks[i+1, 0] = mean

    return blocks


def make_blocks(img_shape, block_shape, overlaps=(0, 36, 36), nonoverlapping_mask=False):
    bz, by, bx = block_shape

    bl_lims = []
    for i in range(3):
        bl_lims.append(make_blocks_1d(
            img_shape[i], block_shape[i], overlaps[i], nonoverlapping_mask=nonoverlapping_mask))

    z_start, y_start, x_start = n.meshgrid(
        *[xx[:, 0] for xx in bl_lims], indexing='ij')
    z_end, y_end, x_end = n.meshgrid(
        *[xx[:, 1] for xx in bl_lims], indexing='ij')

    z_blocks = n.stack([z_start, z_end], axis=-1)
    y_blocks = n.stack([y_start, y_end], axis=-1)
    x_blocks = n.stack([x_start, x_end], axis=-1)

    grid = z_blocks.shape[:-1]
    n_blocks = n.product(grid)
    if n.sum(overlaps) > 0:
        min_grid = make_blocks(img_shape, block_shape, (0, 0, 0))[
            0].shape[1:-1]
        n_min_blocks = n.product(min_grid)
        print("%d blocks with overlap (%d without, %.2fx increase)" % (n_blocks,n_min_blocks, n_blocks/n_min_blocks))


    blocks = n.stack([z_blocks, y_blocks, x_blocks])
    grid_shape = blocks.shape[1:-1]
    blocks = blocks.reshape(3, -1, 2)
    return blocks, grid_shape


def run_detection_on_stack(block_dirs, n_comp=256, temporal_high_pass_width=400, intensity_thresh=None,
                           npil_high_pass_xy=35, npil_high_pass_z=1, detection_unif_filter_xy=3, detection_unif_filter_z=3,
                           norm_by_dif=True, use_hpf_for_norm=True, conv_mode='constant', compute=True,
                           n_xy_chunk=128, n_time_chunk=8000, n_comp_chunk=32, n_z_chunk=4, **kwargs):
    assert intensity_thresh is None
    us, ss, vs, us_raw, vs_raw = load_and_preprocess_stack(block_dirs, n_comp, temporal_high_pass_width,
                                                           norm_by_dif, n_time_chunk,
                                                           use_hpf_for_norm=use_hpf_for_norm)
    us, ss, vs = rechunk_usv_comps(us, ss, vs, comp_chunksize=n_comp_chunk,
                                   z_chunksize=n_z_chunk, t_chunksize=None)
    vs = squareify_vs(vs, chunks=(n_xy_chunk, n_xy_chunk))
    v_sub = npil_sub_v(vs, npil_high_pass_z, npil_high_pass_xy, mode=conv_mode)
    v_conv = conv2d_v(v_sub, detection_unif_filter_z,
                      detection_unif_filter_xy, mode=conv_mode)
    vmap = vmap_from_usv(us, ss, v_conv, rechunk_for_var=True)

    ret = (vmap, us, us_raw, ss, vs_raw, v_sub, v_conv)
    if compute:
        ret = dask.compute(*ret, scheduler='threads')
    return ret


def get_block_idx(zbl, ybl, xbl, grid):
    return n.ravel_multi_index((zbl, ybl, xbl), grid)


def get_stack_idxs(ybl, xbl, grid):
    nz = grid[0]
    idxs = []
    for zbl in range(nz):
        idxs.append(get_block_idx(zbl, ybl, xbl, grid))
    return idxs


def squareify_vs(vs, shape=None, chunks=None):
    if shape is None:
        nyb = int(n.sqrt(vs.shape[-1]))
        assert nyb**2 == vs.shape[-1]
        shape = (nyb, nyb)
    vs = vs.reshape(vs.shape[:-1] + shape)
    if chunks is not None:
        if vs.chunksize[2:] != chunks:
            vs = vs.rechunk((None, None) + chunks)
    return vs


def npil_sub_v(v, npil_filt_z, npil_filt_xy, mode='constant'):
    v = v - conv2d_v(v, npil_filt_z, npil_filt_xy, mode=mode)
    return v


def fix_vmap_edges(vmap, minval=None):
    if minval is None:
        minval = vmap.min()
    vmap[:, -1] = minval
    vmap[:, 0] = minval
    vmap[:, :, -1] = minval
    vmap[:, :, 0] = minval
    return vmap


def conv2d_v(v, conv_filt_z, conv_filt_xy, mode='constant'):
    filt_shape = (1, conv_filt_z, conv_filt_xy, conv_filt_xy)
    v = filter_v(v, filt_shape, norm=True, mode=mode)
    return v


def vmap_from_usv(us, ss, vs, rechunk_for_var=True, intensity_thresh=None):
    assert intensity_thresh is None
    __, nz, nc = us.shape
    __, __, nyb, nxb = vs.shape
    vs_flat = vs.reshape(nc, nz, nyb*nxb)

    vmaps = []
    for zidx in range(nz):
        vmap_i = svd_variance(us[:, zidx], ss[:, zidx], vs_flat[:, zidx],
                              rechunk_comps=rechunk_for_var).reshape(nyb, nxb)
        vmaps.append(vmap_i)
    vmaps = darr.stack(vmaps, axis=0).rechunk((-1, -1, -1))
    return vmaps


def filter_v(v, filter_shape, mode='constant', norm=True):
    if norm:
        filt_mask = dask_uniform_filter(
            darr.ones_like(v), filter_shape, mode=mode)
    v_filt = dask_uniform_filter(v, filter_shape, mode=mode)
    if norm:
        v_filt = v_filt / filt_mask
    return v_filt


def rechunk_usv_comps(u, s, v, comp_chunksize=16, z_chunksize=None, t_chunksize=None):
    # works for single u,s,v or a z-stack along axis 1
    # works for square-ified and flattened vs
    if u.chunksize[-1] != comp_chunksize or z_chunksize is not None:
        cs = list(u.chunksize)
        cs[0] = t_chunksize
        cs[-1] = comp_chunksize
        if z_chunksize is not None:
            cs[1] = z_chunksize
        u = u.rechunk(tuple(cs))
    if s.chunksize[0] != comp_chunksize or z_chunksize is not None:
        cs = list(s.chunksize)
        cs[0] = comp_chunksize
        if z_chunksize is not None:
            cs[1] = z_chunksize
        s = s.rechunk(tuple(cs))
    if v.chunksize[0] != comp_chunksize or z_chunksize is not None:
        cs = list(v.chunksize)
        cs[0] = comp_chunksize
        if z_chunksize is not None:
            cs[1] = z_chunksize
        v = v.rechunk(tuple(cs))
    return u, s, v

def reconstruct_movie_batch(stack_block_dirs, svs, t_indices, vol_shape, block_limits, us=None, log_cb=default_log):
    block_shape = tuple(n.diff(block_limits[:,0])[:,0])
    block_overlaps = tuple([int(n.median(block_shape[i] - n.diff(n.unique(block_limits[i,:,0]))))\
                            for i in range(3)])
    nt = t_indices[1]-t_indices[0]
    n_block, n_comp, n_pix = svs.shape

    len(stack_block_dirs)
    if us is None: 
        us = load_stack_us(stack_block_dirs, n_comp=n_comp, t_indices=t_indices, compute=True)
        log_cb("Loaded Us", 3)

    mask = get_overlap_mask(block_shape, block_overlaps)
    mov_out = n.zeros((nt,) + vol_shape, dtype=n.float32)
    norm = n.zeros(vol_shape, dtype=n.float32)
    for i in range(n_block):
        if i % 25 == 0: log_cb("Reconstructing block %d" % i, 4)
        zz,yy,xx = block_limits[:,i]
        mov_out[:, zz[0]:zz[1], yy[0]:yy[1], xx[0]:xx[1]] += mask * (us[i] @ svs[i]).reshape(nt, *block_shape)
        norm[zz[0]:zz[1], yy[0]:yy[1], xx[0]:xx[1]] += mask
    norm[norm < 1e-4] = n.inf
    return mov_out / norm


def load_and_multiply_stack_svs(stack_block_dirs, n_comp, compute=True):
    ss, vs = load_stack_svs(stack_block_dirs, n_comp, compute=False)
    # print(ss.dtype)
    svs = (vs * ss[:,:,n.newaxis])
    # print(svs.dtype)
    if compute: svs = svs.compute()
    return svs

def load_stack_svs(stack_block_dirs, n_comp, compute=False):
    ss = []; vs = []
    n_blocks = len(stack_block_dirs)
    for block_dir in stack_block_dirs:
        # somehow it is faster to combine them all then compute everything after
        s,v = load_sv(block_dir, n_comp, compute=False)
        ss.append(s)
        vs.append(v)
    ss = darr.array(ss, dtype=n.float32)
    vs = darr.array(vs, dtype=n.float32)
    if compute:
        ss = ss.compute()
        vs = vs.compute()
    return ss, vs

def load_stack_us(stack_block_dirs, n_comp, t_indices=None, compute=False):
    us = []
    for block_dir in stack_block_dirs:
        u = load_u(block_dir, n_comp, t_indices=t_indices, compute=False)
        us.append(u)
    us = darr.array(us)
    if compute: us = us.compute()
    return us

def load_stack_usvs(stack_block_dirs, n_comp, stack_axis=1, compute=False,t_batch_size=None, t_indices=None):
    us = []
    ss = []
    vs = []
    for block_dir in stack_block_dirs:
        u, s, v = load_usv(block_dir, n_comp, t_indices=t_indices)
        us.append(u)
        ss.append(s)
        vs.append(v)
    us = darr.stack(us, axis=stack_axis)
    vs = darr.stack(vs, axis=stack_axis)
    ss = darr.stack(ss, axis=stack_axis)
    if compute:
        us = us.compute()
        vs = vs.compute()
        ss = ss.compute()
    return us, ss, vs


def load_and_preprocess_stack(stack_block_dirs, n_comp, temporal_hpf_width=400, norm_by_dif=True, out_t_chunk=None,
                              use_hpf_for_norm=True):
    us = []
    ss = []
    vs = []
    us_raw = []
    vs_raw = []
    for zbl, block_dir in enumerate(stack_block_dirs):
        u, s, v = load_usv(block_dir, n_comp, t_chunks=temporal_hpf_width)
        us_raw.append(u)
        vs_raw.append(v)
        u, s, v = hpf_and_norm(u, s, v, window_size=temporal_hpf_width,
                               norm_by_dif=norm_by_dif, out_t_chunk=out_t_chunk,
                               use_hpf_u_for_norm=use_hpf_for_norm)
        us.append(u)
        ss.append(s)
        vs.append(v)

    us = darr.stack(us, axis=1)
    us_raw = darr.stack(us_raw, axis=1)
    vs = darr.stack(vs, axis=1)
    vs_raw = darr.stack(vs_raw, axis=1)
    ss = darr.stack(ss, axis=1)
    return us, ss, vs, us_raw, vs_raw


def temporal_hpf(u, window_size, out_t_chunk=None):
    if u.chunksize[0] != window_size:
        print("Rechunking to do HPF - this leads to bad performance. Chunk u correctly before passing it here")
        u = u.rechunk((window_size, None))
    u_hpf = u.map_blocks(lambda x: x - x.mean(axis=0))

    if out_t_chunk is not None and out_t_chunk != u.chunksize[0]:
        u_hpf = u_hpf.rechunk((out_t_chunk, None))
    return u_hpf


def hpf_and_norm(u, s, v, window_size, norm_by_dif=True, out_t_chunk=None, use_hpf_u_for_norm=True):
    u_hpf = temporal_hpf(u, window_size=window_size, out_t_chunk=out_t_chunk)
    if use_hpf_u_for_norm:
        u = u_hpf
    if norm_by_dif:
        ud = darr.diff(u, axis=0, n=1)
        var = svd_variance(ud, s, v)
    else:
        var = svd_variance(u, s, v)
    v = v / var
    return u_hpf, s, v


def load_usv(block_dir, n_comp, t_chunks=None, v_chunks=-1, compute=False, t_indices=None, dtype=n.float32):
    u = darr.from_zarr(os.path.join(block_dir, 'u.zarr'), chunks=(-1, n_comp))[:, :n_comp].astype(dtype)
    if t_chunks is not None:
        u = u.rechunk((t_chunks, -1))
    if t_indices is not None:
        u = u[t_indices[0]:t_indices[1]]
    s = darr.from_zarr(os.path.join(block_dir, 's.zarr'), chunks=(n_comp))[:n_comp].astype(dtype)
    v = darr.from_zarr(os.path.join(block_dir, 'v.zarr'), chunks=(n_comp, -1))[:n_comp].astype(dtype)
    if v_chunks is not None:
        v = v.rechunk((None, v_chunks))
    if compute:
        return u.compute(), s.compute(), v.compute()
    return u, s, v

def load_u(block_dir, n_comp,compute=False, t_indices = None, t_chunks = -1, dtype=n.float32):
    if t_indices is not None and t_chunks == -1: t_chunks = t_indices[1]-t_indices[0]
    u = darr.from_zarr(os.path.join(block_dir, 'u.zarr'),
                       chunks=(t_chunks, n_comp))[:, :n_comp].astype(dtype)
    if t_indices is not None:
        u = u[t_indices[0]:t_indices[1]]
    if compute:
        return u.compute()
    else:
        return u


def load_sv(block_dir, n_comp, compute=False, v_chunks=-1, dtype=n.float32):
    s = darr.from_zarr(os.path.join(block_dir, 's.zarr'), chunks=(n_comp))[:n_comp].astype(dtype)
    v = darr.from_zarr(os.path.join(block_dir, 'v.zarr'), chunks=(n_comp, v_chunks))[:n_comp].astype(dtype)
    if compute:
        return s.compute(), v.compute()
    return s, v


def load_usv2(block_dir, n_comp, t_chunks=None, v_chunks=-1):
    if t_chunks is not None:
        u = darr.from_zarr(os.path.join(block_dir, 'u.zarr'), chunks=(t_chunks, n_comp))[:, :n_comp]
    s = darr.from_zarr(os.path.join(block_dir, 's.zarr'))[:n_comp]
    v = darr.from_zarr(os.path.join(block_dir, 'v.zarr'))[:n_comp]
    if v_chunks is not None:
        v = darr.from_zarr(os.path.join(block_dir, 'v.zarr'), chunks=(n_comp, v_chunks))[:n_comp]
    return u, s, v


def svd_variance(u, s, v, dask=True, n_comp=None, nt=None, sqrt=True, rechunk_comps=False, mean_subtract=False):

    if nt is None:
        nt = u.shape[0]
    if n_comp is None:
        n_comp = s.shape[0]
    if rechunk_comps and dask:
        u = u.rechunk((-1, n_comp))
        v = v.rechunk((n_comp, -1))
        s = s.rechunk(n_comp)
    if mean_subtract:
        print(u.shape)
        print(u.mean(axis=0).shape)
        u = u - u.mean(axis=0)

    if dask:
        sdiag = darr.diag(s[:n_comp])
    else:
        sdiag = n.diag(s[:n_comp])

    C = sdiag @ u.T @ u @ sdiag
    vc = v.T @ C
    var = ((vc * v.T)[:, :n_comp].sum(axis=1)) / nt
    if sqrt:
        var = n.sqrt(var)
    return var
