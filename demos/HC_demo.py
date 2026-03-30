"""
HC_demo: Suite3D pipeline with conservative/default parameters.

Uses standard defaults suitable for typical 2-photon data. Good starting
point for new datasets. Diagnostic figures are saved after each step.

For experimental parameters with quality metrics and sweep support, see v1_demo.py.

Usage:
    python demos/HC_demo.py --data_dir /path/to/tifs --output_dir /path/to/output

    # Re-run from existing job (skip init/registration):
    python demos/HC_demo.py --output_dir /path/to/output --job_id my_job --skip_init --skip_register
"""

import argparse
import os
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from suite3d.job import Job
from suite3d import io


def get_params(tifs):
    """Return parameters for a standard multi-plane 2P recording.

    Adjust these for your specific microscope and experiment setup.
    """
    params = {
        # Volume rate (Hz) - automatically extracted from ScanImage metadata
        "fs": io.get_vol_rate(tifs[0]),

        # GCaMP decay time in seconds (1.3 for GCaMP6s, ~0.7 for GCaMP8f)
        "tau": 1.3,

        # Planes to analyze. Typically exclude the flyback plane (plane 0).
        "planes": np.array([1, 2, 3, 4]),

        # Total number of planes per volume in the TIFF (including flyback)
        "n_ch_tif": 5,

        # Voxel size in microns (z, y, x)
        "voxel_size_um": (20, 1.5, 1.5),

        # Not LBM data
        "lbm": False,
        "subtract_crosstalk": False,
        "fuse_strips": False,

        # Color channels (set num_colors > 1 if you recorded multiple channels)
        "num_colors": 2,
        "functional_color_channel": 0,

        # Number of TIFFs for initialization (~500 frames is usually enough)
        "n_init_files": 2,

        # Registration settings (set gpu_reg=True if you have cupy installed)
        "3d_reg": True,
        "gpu_reg": True,
    }

    # Auto-detect GPU availability and fall back to CPU if needed
    try:
        import cupy  # noqa: F401
    except ImportError:
        params["gpu_reg"] = False
        params["3d_reg"] = False
        print("Note: cupy not found, falling back to CPU registration")

    # Split large registered TIFFs into smaller files
    params["split_tif_size"] = 100

    return params


# =============================================================================
# Diagnostic figure generation
# =============================================================================

def save_init_figures(job, fig_dir):
    """Save initialization diagnostic figures."""
    summary = job.load_summary()
    ref_img = summary["ref_img_3d"]
    nz = ref_img.shape[0]

    # Reference image montage (all planes)
    ncols = min(nz, 5)
    nrows = int(np.ceil(nz / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    for z in range(nz):
        ax = axes[z // ncols, z % ncols]
        ax.imshow(ref_img[z], cmap="gray", vmin=np.percentile(ref_img[z], 1),
                  vmax=np.percentile(ref_img[z], 99.5))
        ax.set_title(f"Plane {z}")
        ax.axis("off")
    for z in range(nz, nrows * ncols):
        axes[z // ncols, z % ncols].axis("off")
    fig.suptitle("Reference Image (all planes)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "01_reference_image.png"), dpi=150)
    plt.close(fig)

    # Plane shifts
    if "plane_shifts" in summary:
        fig, ax = plt.subplots(figsize=(6, 3))
        shifts = summary["plane_shifts"]
        ax.plot(shifts[:, 0], "o-", label="Y shift")
        ax.plot(shifts[:, 1], "s-", label="X shift")
        ax.set_xlabel("Plane")
        ax.set_ylabel("Shift (pixels)")
        ax.set_title("Inter-plane alignment shifts")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "02_plane_shifts.png"), dpi=150)
        plt.close(fig)

    print(f"  Saved init figures to {fig_dir}")


def save_registration_figures(job, fig_dir):
    """Save registration quality diagnostic figures."""
    try:
        reg_results = job.load_registration_results()
    except Exception:
        print("  Could not load registration results for figures")
        return

    if reg_results is None:
        return

    # 3D registration stores int_shift (nt, 3) = z,y,x shifts
    shifts = reg_results.get("int_shift")
    sub_shifts = reg_results.get("sub_pixel_shifts", shifts)

    if sub_shifts is not None:
        sub_shifts = np.array(sub_shifts) if not isinstance(sub_shifts, np.ndarray) else sub_shifts
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        labels = ["Z shift", "Y shift", "X shift"]
        for dim in range(min(sub_shifts.shape[1], 3)):
            axes[dim].plot(sub_shifts[:, dim], alpha=0.7, linewidth=0.5)
            axes[dim].set_ylabel(f"{labels[dim]} (px)")
        axes[0].set_title("Registration shifts over time")
        axes[-1].set_xlabel("Frame")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "03_registration_offsets.png"), dpi=150)
        plt.close(fig)

    # Metrics: list of per-batch arrays or concatenated
    metrics = reg_results.get("metrics")
    if metrics is not None:
        if isinstance(metrics, list):
            # Flatten list of arrays
            flat_metrics = []
            for m in metrics:
                if isinstance(m, np.ndarray) and m.ndim >= 1:
                    flat_metrics.append(m)
            if flat_metrics:
                metrics = np.concatenate(flat_metrics)
            else:
                metrics = None
        if isinstance(metrics, np.ndarray) and metrics.size > 0:
            fig, ax = plt.subplots(figsize=(12, 3))
            if metrics.ndim == 2:
                for z in range(min(metrics.shape[1], 4)):
                    ax.plot(metrics[:, z], alpha=0.7, label=f"z={z}")
                ax.legend(fontsize=8)
            else:
                ax.plot(metrics)
            ax.set_ylabel("Registration quality")
            ax.set_xlabel("Frame")
            ax.set_title("Registration quality metric over time")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "04_registration_quality.png"), dpi=150)
            plt.close(fig)

    print(f"  Saved registration figures to {fig_dir}")


def save_corrmap_figures(job, fig_dir):
    """Save correlation map diagnostic figures."""
    res = job.load_corr_map_results()
    vmap = res.get("vmap")
    mean_img = res.get("mean_img")
    max_img = res.get("max_img")

    nz = vmap.shape[0] if vmap is not None else 0
    ncols = min(nz, 5)
    nrows = int(np.ceil(nz / ncols)) if nz > 0 else 1

    # Correlation map montage
    if vmap is not None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_2d(axes)
        for z in range(nz):
            ax = axes[z // ncols, z % ncols]
            ax.imshow(vmap[z], cmap="hot", vmin=0,
                      vmax=np.percentile(vmap[vmap > 0], 99) if (vmap > 0).any() else 1)
            ax.set_title(f"Plane {z}")
            ax.axis("off")
        for z in range(nz, nrows * ncols):
            axes[z // ncols, z % ncols].axis("off")
        fig.suptitle("Correlation Map (all planes)", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "05_correlation_map.png"), dpi=150)
        plt.close(fig)

    # Mean image montage
    if mean_img is not None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_2d(axes)
        for z in range(nz):
            ax = axes[z // ncols, z % ncols]
            ax.imshow(mean_img[z], cmap="gray", vmin=np.percentile(mean_img[z], 1),
                      vmax=np.percentile(mean_img[z], 99.5))
            ax.set_title(f"Plane {z}")
            ax.axis("off")
        for z in range(nz, nrows * ncols):
            axes[z // ncols, z % ncols].axis("off")
        fig.suptitle("Mean Image (all planes)", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "06_mean_image.png"), dpi=150)
        plt.close(fig)

    print(f"  Saved corrmap figures to {fig_dir}")


def save_segmentation_figures(job, fig_dir):
    """Save cell detection diagnostic figures."""
    try:
        seg_results = job.load_segmentation_results()
    except Exception:
        print("  Could not load segmentation results for figures")
        return

    stats = seg_results.get("stats", seg_results) if isinstance(seg_results, dict) else seg_results
    if not hasattr(stats, '__len__'):
        return

    res = job.load_corr_map_results()
    vmap = res.get("vmap")
    if vmap is None:
        return

    nz = vmap.shape[0]
    ncols = min(nz, 5)
    nrows = int(np.ceil(nz / ncols))

    # Cell masks overlaid on correlation map
    cell_masks = np.zeros_like(vmap)
    for stat in stats:
        if "coords" in stat and "lam" in stat:
            cz, cy, cx = stat["coords"]
            lam = stat["lam"]
            cell_masks[cz, cy, cx] = np.maximum(cell_masks[cz, cy, cx], lam / lam.max())

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    for z in range(nz):
        ax = axes[z // ncols, z % ncols]
        # Show corrmap as gray background
        vmax = np.percentile(vmap[vmap > 0], 99) if (vmap > 0).any() else 1
        ax.imshow(vmap[z], cmap="gray", vmin=0, vmax=vmax)
        # Overlay cell masks in red
        mask_z = cell_masks[z]
        if mask_z.max() > 0:
            overlay = np.zeros((*mask_z.shape, 4))
            overlay[..., 0] = 1.0  # red channel
            overlay[..., 3] = mask_z / mask_z.max() * 0.6  # alpha
            ax.imshow(overlay)
        ax.set_title(f"Plane {z} ({(cell_masks[z] > 0).sum()} px)")
        ax.axis("off")
    for z in range(nz, nrows * ncols):
        axes[z // ncols, z % ncols].axis("off")
    fig.suptitle(f"Detected cells ({len(stats)} ROIs) overlaid on correlation map", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "07_cell_masks.png"), dpi=150)
    plt.close(fig)

    # Cell statistics
    n_voxels = [len(s["lam"]) for s in stats if "lam" in s]
    if n_voxels:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(n_voxels, bins=50, edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Voxels per ROI")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"ROI size distribution (n={len(stats)})")

        # Z-plane distribution
        z_meds = [s["med"][0] for s in stats if "med" in s]
        if z_meds:
            axes[1].hist(z_meds, bins=np.arange(nz + 1) - 0.5, edgecolor="black", alpha=0.7)
            axes[1].set_xlabel("Z plane")
            axes[1].set_ylabel("Count")
            axes[1].set_title("ROIs per z-plane")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "08_cell_statistics.png"), dpi=150)
        plt.close(fig)

    print(f"  Saved segmentation figures to {fig_dir}")


# =============================================================================
# Pipeline
# =============================================================================

def run_pipeline(data_dir, output_dir, job_id, skip_init, skip_register,
                 skip_corrmap, skip_segment, skip_extract, export_dir):
    """Run the suite3d processing pipeline."""

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

    # Create figures directory
    fig_dir = os.path.join(job.job_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Run pipeline stages
    if not skip_init:
        print("\n=== Running initialization ===")
        job.run_init_pass()
        save_init_figures(job, fig_dir)

    if not skip_register:
        print("\n=== Running registration ===")
        job.register()
        save_registration_figures(job, fig_dir)

    if not skip_corrmap:
        print("\n=== Calculating correlation map ===")
        job.calculate_corr_map()
        save_corrmap_figures(job, fig_dir)

    if not skip_segment:
        print("\n=== Segmenting ROIs ===")
        job.segment_rois()
        save_segmentation_figures(job, fig_dir)

    if not skip_extract:
        print("\n=== Computing neuropil masks and extracting traces ===")
        job.compute_npil_masks()
        job.extract_and_deconvolve()

    if export_dir:
        print(f"\n=== Exporting results to {export_dir} ===")
        job.export_results(str(export_dir), result_dir_name="rois")

    print(f"\nPipeline complete! Figures saved to {fig_dir}")
    return job


def main():
    parser = argparse.ArgumentParser(
        description="Suite3D pipeline for standard multi-plane 2P data"
    )
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing raw ScanImage TIFF files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store suite3d job outputs")
    parser.add_argument("--job_id", type=str, default="demo-std",
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
