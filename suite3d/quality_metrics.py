import numpy as n
import numba


def volume_quality(volume, pct_high = 99.99, pct_low = 25.0):
    '''
    return some heuristics about the signal level in the mean volume provided

    Args:
        volume (ndarray): nz,ny,nx - averaged movie over time
        pct_high (float, optional): high percentile to take as the peak of 'signal' . Defaults to 99.99.
        pct_low (float, optional): low percentile to take as 'background'. Defaults to 25.0.

    Returns:
        metrics: dictionary
    '''
    sig = n.percentile(volume, pct_high, axis=(1,2))
    bg = n.percentile(volume, pct_low, axis=(1,2))


    metrics = {
        'signal_range':  sig - bg,
        'signal_to_background_ratio' : sig / bg,
        'mean_fluorescence' : volume.mean(axis=(1,2)),
        'volume_std'  : volume.std(axis=(1,2)),
    }
    return metrics

def shot_noise_pct(fs, frate_hz):
    '''
    compute the theoretical shot noise percentage in a timeseries 
    assumes GCamP6s, using equation from Pachitariu

    Args:
        fs (ndarray): npix, nt - timeseries
        frate_hz (float): frame rate

    Returns:
        noise_level: array of percentage noise for each pixel
    '''
    df = n.diff(fs, axis=1)
    dff = df / fs.mean(axis=1,keepdims=True)
    abs_d_dff = n.abs(n.diff(dff,axis=1))
    noise_level = n.nanmedian(abs_d_dff, axis=1)
    noise_level = noise_level / frate_hz

    return noise_level

@numba.jit(nopython=True, parallel=True)
def shot_noise_suyash(fs, frate_hz):
    npix, nt = fs.shape
    noise_level = n.empty(npix)
    
    for i in numba.prange(npix):
        row = fs[i]
        mean_f = n.mean(row)
        df = n.diff(row) / mean_f
        noise_level[i] = n.nanmedian(n.abs(n.diff(df)))
    
    return noise_level / frate_hz

def choose_top_pix(vol, pct = 98):
    nz, ny, nx = vol.shape
    pcts = n.array([n.percentile(vol[i].flatten(), pct)  for i in range(nz)])
    n_top_pix = int((100-pct) * ny * nx / 100 - 2)
    top_pix = n.array([vol[i] >= pcts[i] for i in range(nz)])
    return top_pix

def compute_metrics_for_movie(mov, frate_hz, top_pix=None):
    nz, nt, ny, nx = mov.shape
    vol = mov.mean(axis=1)
    metrics = volume_quality(vol)
    if top_pix is None:
        top_pix = choose_top_pix(vol)

    noises = []
    npix = (top_pix.sum(axis=(1,2))).min()
    # print(npix)
    for i in range(nz):
        noises.append(shot_noise_pct(mov[i][:,top_pix[i]][:,:npix].reshape(nt,-1).T, frate_hz))
        # print(noises[-1].shape)
    noise_levels = n.array(noises)

    metrics['noise_levels'] = noise_levels

    return vol, metrics

    
    
