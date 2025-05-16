from pathlib import Path
import os
import numpy as np
from suite3d.job import Job
# from suite3d import ui
# from suite3d import io

os.chdir(os.path.dirname(os.path.abspath("")))


def main(load=False):

    fpath = Path(r"D://demo")
    job_path = fpath.joinpath("results")

    # Set the mandatory parameters
    params = {
        # volume rate
        'fs': 17,
        'planes': np.arange(14),
        'n_ch_tif': 14,
        'tau': 1.3,
        'lbm': True,
        'fuse_strips': True,
        'subtract_crosstalk': False,
        'init_n_frames': None,
        'n_init_files': 1,
        'n_proc_corr': 12,
        'max_rigid_shift_pix': 150,
        '3d_reg': True,
        'gpu_reg': True,
        'block_size': [64, 64],
    }

    # Set the optional parameters
    if load:
        job = Job(str(job_path), 'v1', create=False, overwrite=False)
        return job
    else:
        tifs = list(fpath.joinpath("raw").glob("*.tif*"))
        job = Job(str(job_path), 'v1', create=True, overwrite=True, verbosity = 1, tifs=tifs, params=params)
        job.run_init_pass()
        job.register()
        job.calculate_corr_map()
        job.params['patch_size_xy'] = (250, 250)
        job.segment_rois()
        return job


if __name__ == '__main__':
    job = main(load=True)
    job.register()
    # corr = job.calculate_corr_map()
    # job.params['patch_size_xy'] = (250, 250)
    # job.segment_rois(patches_to_segment=(5,))
    x=2