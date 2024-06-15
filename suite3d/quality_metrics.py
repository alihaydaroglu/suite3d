import numpy as n


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


def compute_metrics_for_movie(mov, time_axis=1):
    vol = mov.mean(axis=time_axis)