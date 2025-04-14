import os
from pathlib import Path

import click
import numpy as np
import napari
from suite3d.job import Job
from suite3d import io

os.chdir(os.path.dirname(os.path.abspath("")))
os.environ["SUITE3D_DEVELOPER"] = "true"

def get_params(tifs):
    return {
        'fs': io.get_vol_rate(tifs[0]),
        'cavity_size': 1,
        'planes': np.arange(14),
        'n_ch_tif': 14,
        'max_reg_xy_reference': 140,
        'tau': 1.3,
        'lbm': True,
        'fuse_strips': True,
        'subtract_crosstalk': False,
        '3d_reg': True,
        'gpu_reg': True,
        'voxel_size_um': (17, 2, 2),
        'intensity_thresh': 0.1,
        'block_size': [128, 128],
    }

def get_job(base_dir):
    fpath = Path(base_dir)
    job_path = fpath.joinpath("results")
    tif_path = fpath.joinpath("raw")
    tifs = io.get_tif_paths(str(tif_path))
    params = get_params(tifs)
    return Job(job_path, 'v1', create=True, overwrite=True, verbosity=3, tifs=tifs, params=params)

def run_job(job):
    job.run_init_pass()
    job.register()
    return job.get_registered_movie()

def view_data(im_full):
    crop = ((0, 13), (130, 445), (10, 440))
    cropped = im_full[:, :, crop[1][0]:crop[1][1], crop[2][0]:crop[2][1]]
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(cropped, name='Imaging Data', scale=(1, 8, 1, 1), rendering='mip')
    napari.run()

@click.command()
@click.option(
    '--job-dir',
    prompt='Base directory for a job, not including the `job_id` string.',
    help='Base directory to hold jobs by job_id.'
)
@click.option(
    '--job-id',
    prompt='Directory ID to name this job.',
    default='TEST',
    help='Job ID to load or create.'
)
def main(job_dir):
    job = get_job(job_dir)
    im_full = run_job(job)
    view_data(im_full)

if __name__ == "__main__":
    main()
