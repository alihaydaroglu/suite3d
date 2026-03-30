"""
Suite3D demo script for Light Beads Microscopy (LBM) data.

LBM data has specific characteristics: multiple ROI strips that need fusing,
crosstalk between cavities that needs subtraction, and plane reordering
from ScanImage temporal order to depth order.

Usage:
    python demos/demo_lbm.py --data_dir /path/to/tifs --output_dir /path/to/output

    # To re-run from an existing job (skipping init/registration):
    python demos/demo_lbm.py --output_dir /path/to/output --job_id my_job --skip_init --skip_register
"""

import argparse
import numpy as np
from pathlib import Path

from suite3d.job import Job
from suite3d import io


def get_params(tifs):
    """Return parameters for an LBM recording.

    Adjust these for your specific microscope setup. The key LBM-specific
    parameters are: planes (reordering), cavity_size, fuse_shift_override,
    and subtract_crosstalk.
    """
    params = {
        # Volume rate (Hz) - extracted from ScanImage metadata
        "fs": io.get_si_params(tifs[0])["vol_rate"],

        # GCaMP decay time (seconds)
        "tau": 1.3,

        # Number of channels per volume in the TIFF
        "n_ch_tif": 26,

        # Number of planes per cavity (for crosstalk subtraction)
        "cavity_size": 13,

        # Plane ordering: convert ScanImage temporal order to depth order.
        # This mapping is specific to your microscope configuration.
        "planes": np.array([
            0, 2, 4, 6, 8, 10, 12, 14,
            16, 18, 20, 22, 24, 1, 3, 5, 7, 9, 11, 13, 15, 17,
        ]),

        # Voxel size in microns (z, y, x)
        "voxel_size_um": (20, 2.5, 2.5),

        # Number of TIFFs for initialization (~500 frames is usually enough)
        "n_init_files": 2,

        # LBM-specific: fuse mesoscope strips
        "fuse_strips": True,
        "fuse_shift_override": 7,

        # Crosstalk subtraction between cavities
        "subtract_crosstalk": True,

        # Registration settings
        "3d_reg": True,
        "gpu_reg": True,

        # Split large registered TIFFs into smaller files
        "split_tif_size": 100,
    }
    return params


def run_pipeline(data_dir, output_dir, job_id, skip_init, skip_register,
                 skip_corrmap, skip_segment, skip_extract, export_dir):
    """Run the suite3d processing pipeline for LBM data."""

    data_dir = Path(data_dir) if data_dir else None
    output_dir = Path(output_dir)

    # Load or create job
    job_dir = output_dir / f"s3d-{job_id}"
    if job_dir.exists() and (job_dir / "params.npy").exists():
        print(f"Loading existing job from {job_dir}")
        job = Job(str(output_dir), job_id, create=False)
    else:
        if data_dir is None:
            raise ValueError("--data_dir is required when creating a new job")
        tifs = io.get_tif_paths(str(data_dir))
        if not tifs:
            raise FileNotFoundError(f"No TIFF files found in {data_dir}")
        print(f"Found {len(tifs)} TIFFs in {data_dir}")
        params = get_params(tifs)
        job = Job(str(output_dir), job_id, tifs=tifs, params=params,
                  create=True, overwrite=True, verbosity=3)

    # Run pipeline stages
    if not skip_init:
        print("\n=== Running initialization ===")
        job.run_init_pass()

    if not skip_register:
        print("\n=== Running registration ===")
        job.register()

    if not skip_corrmap:
        print("\n=== Calculating correlation map ===")
        job.calculate_corr_map()

    if not skip_segment:
        print("\n=== Segmenting ROIs ===")
        job.segment_rois()

    if not skip_extract:
        print("\n=== Computing neuropil masks and extracting traces ===")
        job.compute_npil_masks()
        job.extract_and_deconvolve()

    if export_dir:
        print(f"\n=== Exporting results to {export_dir} ===")
        job.export_results(str(export_dir), result_dir_name="rois")

    print("\nPipeline complete!")
    return job


def main():
    parser = argparse.ArgumentParser(
        description="Suite3D pipeline for LBM data"
    )
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing raw LBM TIFF files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store suite3d job outputs")
    parser.add_argument("--job_id", type=str, default="demo-lbm",
                        help="Unique name for this processing job")
    parser.add_argument("--export_dir", type=str, default=None,
                        help="Directory to export final results (optional)")
    parser.add_argument("--skip_init", action="store_true",
                        help="Skip initialization pass")
    parser.add_argument("--skip_register", action="store_true",
                        help="Skip registration")
    parser.add_argument("--skip_corrmap", action="store_true",
                        help="Skip correlation map calculation")
    parser.add_argument("--skip_segment", action="store_true",
                        help="Skip ROI segmentation")
    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip trace extraction and deconvolution")

    args = parser.parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        job_id=args.job_id,
        skip_init=args.skip_init,
        skip_register=args.skip_register,
        skip_corrmap=args.skip_corrmap,
        skip_segment=args.skip_segment,
        skip_extract=args.skip_extract,
        export_dir=args.export_dir,
    )


if __name__ == "__main__":
    main()
