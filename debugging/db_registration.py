from pathlib import Path
import os
import numpy as np

from matplotlib import pyplot as plt
import napari
from suite3d.job import Job
from suite3d import ui
from suite3d import io
os.chdir(os.path.dirname(os.path.abspath("")))


def main():

    fpath = Path(r"D:\W2_DATA\kbarber\2025_03_01\mk301")
    job_path = fpath.joinpath("results")

    # Set the mandatory parameters
    params = {
        # volume rate
        'fs': 17,
        # planes to analyze. 0 is deepest, 30 is shallowest (corrected for ScanImage channel IDs)
        # you should keep all the planes to do crosstalk estimation!
        'planes': np.arange(14),
        'n_ch_tif': 14,
        # 'crosstalk_n_planes': len(planes) // 2,
        # Decay time of the Ca indicator in seconds. 1.3 for GCaMP6s. This example is for GCamP8m
        'tau': 1.3,
        'lbm': True,
        # 'num_colors': 2,  # if not lbm data, how many color channels were recorded by scanimage
        # 'functional_color_channel': 0,  # if not lbm data, which color channel is the functional one
        'fuse_strips': True,  # don't do this, it's only needed for LBM data
        # 'fix_shallow_plane_shift_estimates': True,
        'subtract_crosstalk': False,  # I think this is unnecessary for non LBM data...
        'init_n_frames': None,
        'n_init_files': 1,
        'n_proc_corr': 12,
        'max_rigid_shift_pix': 150,
        '3d_reg': True,
        'gpu_reg': True,
        'block_size': [64, 64],
    }

    # set to "false" to run code without messages intended for developers
    os.environ["SUITE3D_DEVELOPER"] = "true"

    tifs = list(fpath.joinpath("green").glob("*.tif*"))
    job = Job(job_path, 'v1', create=True, overwrite=True, verbosity = 2, tifs=tifs, params=params)
    # job.run_init_pass()
    # job.register_gpu()
    # job.register()
    # job.params['n_skip'] = job.load_summary()['fuse_shift']
    # job.fuse_registered_movie()
    return job


if __name__ == '__main__':
    job = main()
    print(job)
    x=2