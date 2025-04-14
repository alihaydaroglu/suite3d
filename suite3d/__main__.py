import os
from pathlib import Path
import click
import numpy as np
import napari
from suite3d.job import Job
from suite3d import io

os.chdir(os.path.dirname(os.path.abspath("")))
os.environ["SUITE3D_DEVELOPER"] = "true"

def get_params():

    # basic imaging
    params = {
        "tau": 1.3,                      # GCamp6s parameter (example)
        "voxel_size_um": (17, 2, 2),     # size of a voxel in microns (z, y, x)
        "planes": np.arange(14),         # planes to analyze (0-based indexing)
        "n_ch_tif": 14,                  # number of channels/planes in each TIFF
    }

    # Filtering Parameters (Cell detection & Neuropil subtraction)
    params.update({
        "cell_filt_type": "gaussian",  # cell detection filter type
        "npil_filt_type": "gaussian",  # neuropil filter type
        "cell_filt_xy_um": 5.0,        # cell detection filter size in xy (microns)
        "npil_filt_xy_um": 3.0,        # neuropil filter size in xy (microns)
        "cell_filt_z_um": 18,          # cell detection filter size in z (microns)
        "npil_filt_z": 2.5,            # neuropil filter size in z (microns)
    })

    # Normalization & Thresholding
    params.update({
        "sdnorm_exp": 0.8,          # normalization exponent for correlation map
        "intensity_thresh": 1,      # threshold for the normalized, filtered movie
    })

    # Compute & Batch Parameters
    params.update({
        "t_batch_size": 300,         # number of frames to compute per iteration
        "n_proc_corr": 12,           # number of processors for correlation map calculation
        "mproc_batchsize": 5,        # frames per smaller batch within the larger batch
        "n_init_files": 1,           # number of TIFFs used for initialization
    })

    # Registration & Advanced Parameters
    params.update({
        "fuse_shift_override": None, # override for fusing shifts if desired
        "init_n_frames": None,       # number of frames to use for initialization (None = use defaults)
        "override_crosstalk": None,  # override for crosstalk subtraction
        "gpu_reg_batchsize": 10,     # batch size for GPU registration
        "max_rigid_shift_pix": 150,  # maximum rigid shift (in pixels) allowed during registration
        "3d_reg": True,              # perform 3D registration
        "gpu_reg": True,             # use GPU acceleration for registration
    })

    return params

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

def get_job(job_dir: str | os.PathLike, job_id: str | os.PathLike, tif_list: str | os.PathLike | None=None):
    """
    Given a directory and a job_id, return a Job object or create a new job if one does not exist.

    Parameters
    ----------
    job_dir : str, os.PathLike
        Path to the directory containing the job-id directory.

    job_id : str, int
        str name for the job, to be appended as f"s3d-{job_id}"

    tif_list : list[str] or list[os.PathLike], optional
        List of paths to raw tifs, needed to create a new job.

    Returns
    -------
    Job
        Object containing parameters, directories and function entrypoints to the pipeline.
    """
    job_dir = Path(job_dir)
    job_path = job_dir / f"s3d-{job_id}"

    # find existing job
    if not job_path.exists():
        print(f"{job_path} does not exist, creating")

        if tif_list:
            print(tif_list)
        else:
            raise ValueError(f"{job_path} does not exist, must provide valid {tif_list}."
                             f"To create a new job, pass tif_list=path/to/tifs")

        # make sure there are valid tifs, and no errors fetching params
        if not isinstance(tif_list, list):
            raise ValueError(f"Argument tif_list should be a list of filepaths, got {type(tif_list)}")
        return Job(job_dir, job_id, create=True, overwrite=True, verbosity=3, tifs=tifs, params=get_params())

    # otherwise, load the job
    return Job(job_dir, job_id, create=False, overwrite=False)


def run_job(job, do_init, do_register, do_correlate, do_detect):
    results = {
        "init": None,
        "register": None,
        "correlate": None,
        "detect": None,
        "registered_movie": None,
    }
    try:
        if do_init:
            job.run_init_pass()
            results["init"] = True
        else:
            results["init"] = False
    except Exception:
        results["init"] = False

    try:
        if do_register:
            job.register()
            results["register"] = True
        else:
            results["register"] = False
    except Exception:
        results["register"] = False

    try:
        if do_correlate:
            job.calculate_corr_map()
            results["correlate"] = True
        else:
            results["correlate"] = False
    except Exception:
        results["correlate"] = False

    try:
        if do_detect:
            job.detect_cells()
            results["detect"] = True
        else:
            results["detect"] = False
    except Exception:
        results["detect"] = False

    return results


def view_data(im_full):
    crop = ((0, 13), (130, 445), (10, 440))
    cropped = im_full[:, :, crop[1][0]:crop[1][1], crop[2][0]:crop[2][1]]
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(cropped, name='Imaging Data', scale=(1, 8, 1, 1), rendering='mip')
    napari.run()

@click.command()
@click.option('--job-dir', prompt='Enter base job directory', help='Base directory to hold jobs.')
@click.option('--job-id', prompt='Enter job ID', default='demo', help='Job ID to load or create.')
@click.option(
    '--tif-dir',
    prompt='Full path to raw ScanImage tiff files (leave empty if loading a job)',
    default=None,
    help='Path to raw ScanImage tifs.'
)
@click.option('--init', is_flag=True, help='Run initialization pass.')
@click.option('--register', is_flag=True, help='Run registration.')
@click.option('--correlate', is_flag=True, help='Calculate correlation map.')
@click.option('--detect', is_flag=True, help='Run detection.')
def main(job_dir, job_id, tif_dir, init, register, correlate, detect):
    job_dir = Path(job_dir).resolve().expanduser()
    if tif_dir is not None:
        tif_list = io.get_tif_paths(tif_dir)
    else:
        tif_list = None

    job = get_job(job_dir, job_id, tif_list)
    job.params.update({"fs": io.get_vol_rate(job.tifs[0])})
    res = run_job(job, init, register, correlate, detect)
    if res:
        print("Success!")
    else:
        print("Failed!")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import pdb
        pdb.post_mortem()
        raise
