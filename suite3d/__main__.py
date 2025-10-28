import os
import traceback
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
        "cavity_size": 1,
        "lbm": True,
    }

    # Filtering Parameters (Cell detection & Neuropil subtraction)
    params.update({
        "cell_filt_type": "gaussian",  # cell detection filter type
        "npil_filt_type": "gaussian",  # neuropil filter type
        "cell_filt_xy_um": 5.0,        # cell detection filter size in xy (microns)
        "npil_filt_xy_um": 3.0,        # neuropil filter size in xy (microns)
        "cell_filt_z_um": 18,          # cell detection filter size in z (microns)
        "npil_filt_z_um": 2.5,            # neuropil filter size in z (microns)
    })

    # Normalization & Thresholding
    params.update({
        "sdnorm_exp": 0.8,          # normalization exponent for correlation map
        "intensity_thresh": 0.7,      # threshold for the normalized, filtered movie
        "extend_thresh": 0.15,
        "detection_timebin": 25,
    })

    # Compute & Batch Parameters
    params.update({
        "t_batch_size": 300,         # number of frames to compute per iteration
        "n_proc_corr": 1,           # number of processors for correlation map calculation
        "mproc_batchsize": 5,        # frames per smaller batch within the larger batch
        "n_init_files": 1,           # number of TIFFs used for initialization
    })

    # Registration
    params.update({
        "fuse_shift_override": None, # override for fusing shifts if desired
        "init_n_frames": None,       # number of frames to use for initialization (None = use defaults)
        "override_crosstalk": None,  # override for crosstalk subtraction
        "gpu_reg_batchsize": 10,     # batch size for GPU registration
        "max_rigid_shift_pix": 250,  # maximum rigid shift (in pixels) allowed during registration
        "max_pix": 2500,             # set this very high and forget about it
        "3d_reg": True,              # perform 3D registration
        "gpu_reg": True,             # use GPU acceleration for registration
    })

    return params


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

    # If tif_list is passed, force recreation
    if tif_list:
        print(f"Forcing new job creation at {job_path}")
        if job_path.exists():
            import shutil
            shutil.rmtree(job_path)
        return Job(job_dir, job_id, create=True, overwrite=True, verbosity=3, tifs=tif_list, params=get_params())

    # Otherwise load existing job
    if not job_path.exists() or not job_path.joinpath("params.npy").exists():
        raise ValueError(f"{job_path} does not exist and no --tif-dir provided to create it.")

    return Job(job_dir, job_id, create=False, overwrite=False)

def run_job(job, do_init, do_register, do_correlate, do_segment):
    results = {
        "init": None,
        "register": None,
        "correlate": None,
        "segment": None,
        "errors": {},
    }

    def run_stage(stage_name, fn):
        try:
            fn()
            results[stage_name] = True
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            location = f"{tb.filename}:{tb.lineno} in {tb.name}"
            results["errors"][stage_name] = f"{type(e).__name__}: {e} (at {location})"
            results[stage_name] = False
            return False
        return True

    if do_init and not run_stage("init", job.run_init_pass):
        return results
    if do_register and not run_stage("register", job.register):
        return results
    if do_correlate and not run_stage("correlate", job.calculate_corr_map):
        return results
    if do_segment and not run_stage("segment", job.segment_rois):
        return results

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
    default='',
    help='Path to raw ScanImage tifs.'
)
@click.option('--init', is_flag=True, help='Run initialization pass.')
@click.option('--register', is_flag=True, help='Run registration.')
@click.option('--correlate', is_flag=True, help='Calculate correlation map.')
@click.option('--segment', is_flag=True, help='Run segmentation.')
@click.option('--all', is_flag=True, help='Run full pipeline.')
def main(job_dir, job_id, tif_dir, init, register, correlate, segment, all):
    job_dir = Path(job_dir).resolve().expanduser()
    if tif_dir:
        tif_list = io.get_tif_paths(tif_dir)
        if not tif_list:
            print(f"No files found in the tif-dir {tif_dir}.")
            return None
    else:
        tif_list = None

    job = get_job(job_dir, job_id, tif_list)
    job.params.update({"fs": io.get_vol_rate(job.tifs[0])})

    if all:
        init, register, correlate, segment = True, True, True, True
    res = run_job(job, init, register, correlate, segment)

    if res["errors"]:
        print("\nErrors occurred:")
        for stage, err in res["errors"].items():
            print(f"- {stage}: {repr(err)}")
    else:
        print("Pipeline ran successfully.")
        print(res)

if __name__ == "__main__":
    main()
