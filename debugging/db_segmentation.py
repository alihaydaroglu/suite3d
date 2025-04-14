from pathlib import Path
import os
import numpy as np

from matplotlib import pyplot as plt
import napari
from suite3d.job import Job
from suite3d import ui
from suite3d import io
from suite3d import tiff_utils as tfu
os.chdir(os.path.dirname(os.path.abspath("")))

def get_vmap(job, npil_filt_xy_um=5, cell_filt_xy_um=5,cell_filt_z_um=15, cft="gaussian"):
    mov_full = job.get_registered_movie('registered_fused_data', 'fused')

    job.params["npil_filt_xy_um"] = npil_filt_xy_um
    job.params["cell_filt_xy_um"] = cell_filt_xy_um
    job.params["cell_filt_z_um"] = cell_filt_z_um
    job.params["cell_filt_type"] = "gaussian"
    job.params["npil_filt_type"] = "gaussian"
    result = job.calculate_corr_map(mov=mov_full)

    print(f"npil filter XY (um): {job.params['npil_filt_xy_um']} ({job.params['npil_filt_type']})")
    print(f"cell filter XY (um): {job.params['cell_filt_xy_um']} ({job.params['cell_filt_type']})")
    print(f"cell filter Z (um): {job.params['cell_filt_z_um']} ({job.params['npil_filt_type']})")

    return result

def main():

    fpath = Path(r"D:\W2_DATA\kbarber\2025_03_01\mk301")
    job_path = fpath.joinpath("results")
    job = Job(job_path, '03_26', create=False, overwrite=False, verbosity=3)
    summary = job.load_summary()
    # mov_full = job.get_registered_movie('registered_fused_data', 'f')
    corr_map = job.load_corr_map_results()['vmap']
    job.segment_rois(vmap=corr_map)



if __name__ == '__main__':
    main()