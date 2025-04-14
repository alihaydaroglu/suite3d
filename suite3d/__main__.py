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

def get_job(job_dir, job_id):
    """
    Given a directory and a job_id, return a Job object or create a new job if one does not exist.

    Parameters
    ----------
    job_dir : str, os.PathLike
        Path to the directory containing the job-id directory.

    job_id : str, int
        str name for the job, to be appended as f"s3d-{job_id}"

    Returns
    -------
    Job
        Object containing parameters, directories and function entrypoints to the pipeline.
    """
    fpath = Path(job_dir)
    job_path = fpath / f"s3d-{job_id}"

    # find existing job
    if not job_path.exists():
        tif_path = fpath.joinpath("raw")
        tifs = io.get_tif_paths(str(tif_path))

        # make sure there are valid tifs, and no errors fetching params
        if not tifs:
            raise FileNotFoundError(f"No tifs found in {str(tif_path)}")
        params = get_params(tifs)
        if not params:
            raise ValueError(f"There was an issue creating params for {str(tif_path)}")
        return Job(job_path, job_id, create=True, overwrite=True, verbosity=3, tifs=tifs, params=params)
    # otherwise, load the job
    return Job(str(job_path), job_id, create=False, overwrite=False)

def run_job(job, do_init, do_register, do_correlate, do_detect):
    if do_init:
        job.run_init_pass()
    if do_register:
        job.register()
    if do_correlate:
        job.calculate_corr_map()
    if do_detect:
        job.detect_cells()
    return job.get_registered_movie()

def view_data(im_full):
    crop = ((0, 13), (130, 445), (10, 440))
    cropped = im_full[:, :, crop[1][0]:crop[1][1], crop[2][0]:crop[2][1]]
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(cropped, name='Imaging Data', scale=(1, 8, 1, 1), rendering='mip')
    napari.run()

@click.command()
@click.option('--job-dir', required=False, help='Base directory to hold jobs.')
@click.option('--job-id', required=False, default='demo', help='Job ID to load or create.')
@click.option('--init', is_flag=True, help='Run initialization pass.')
@click.option('--register', is_flag=True, help='Run registration.')
@click.option('--correlate', is_flag=True, help='Calculate correlation map.')
@click.option('--detect', is_flag=True, help='Run detection.')
def main(job_dir, job_id, init, register, correlate, detect):
    job = get_job(job_dir, job_id)
    im_full = run_job(job, init, register, correlate, detect)
    view_data(im_full)

if __name__ == "__main__":
    main()
