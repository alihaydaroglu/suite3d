"""
v1_demo: Suite3D pipeline with experimental TC030 parameters (March 2025).

Uses aggressive ROI extension (20 iters, 3 dilations/iter), finer spatial
filters (1um xy, 2.5um z), and lower detection thresholds. Includes
quality metric computation (duplication, overmerge, shot noise) and
optional segmentation parameter sweep.

For conservative/default parameters, see HC_demo.py.

Usage:
    python demos/v1_demo.py --data_dir /path/to/tifs --output_dir /path/to/output

    # Re-run from existing job (skip init/registration):
    python demos/v1_demo.py --output_dir /path/to/output --job_id my_job --skip_init --skip_register

    # Run segmentation parameter sweep:
    python demos/v1_demo.py --output_dir /path/to/output --job_id my_job --skip_init --skip_register --skip_corrmap --sweep
"""

import argparse
import os
import time
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from suite3d.job import Job
from suite3d import io


def get_params(tifs):
    """Return parameters for a standard multi-plane 2P recording.

    Uses the experimental parameters validated on TC030 data (March 2025).
    These emphasize aggressive ROI extension and fine spatial filtering
    for high-resolution xy data.
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

        # Color channels
        "num_colors": 2,
        "functional_color_channel": 0,

        # Number of TIFFs for initialization (~500 frames is usually enough)
        "n_init_files": 2,

        # Registration settings
        "3d_reg": True,
        "gpu_reg": True,
        "nonrigid": True,
        "max_shift_nr": (1, 5, 5),
        "nr_npad": (1, 3, 3),
        "block_size_3d": (4, 128, 128),

        # Correlation map - finer spatial filters for high-res data
        "cell_filt_xy_um": 1,
        "cell_filt_z_um": 2.5,
        "detection_timebin": 3,

        # Segmentation - experimental params from TC030 (March 2025)
        "peak_thresh": 0.03,
        "vox_snr_thresh": 0.04,
        "roi_ext_iterations": 20,
        "roi_dilations_per_iter": 3,
        "max_pix": 10000,
        "segmentation_timebin": 2,
        "use_power_iter_v1": True,
        "multi_source": True,
        "ext_subtract_iters": 2,
        "patch_overlap_xy": (75, 75),
        "patch_size_xy": (300, 300),
        "min_frames": 100,
        "segmentation_spatial_filt": 1,
        "n_proc_detect": 32,

        # Split large registered TIFFs into smaller files
        "split_tif_size": 100,
    }

    # Auto-detect GPU availability and fall back to CPU if needed
    try:
        import cupy  # noqa: F401
    except ImportError:
        params["gpu_reg"] = False
        params["3d_reg"] = False
        print("Note: cupy not found, falling back to CPU registration")

    return params


# =============================================================================
# Diagnostic figure generation
# =============================================================================

def save_init_figures(job, fig_dir):
    """Save initialization diagnostic figures."""
    summary = job.load_summary()
    ref_img = summary["ref_img_3d"]
    nz = ref_img.shape[0]

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

    print(f"  Saved registration figures to {fig_dir}")


def save_corrmap_figures(job, fig_dir):
    """Save correlation map diagnostic figures."""
    res = job.load_corr_map_results()
    vmap = res.get("vmap")
    mean_img = res.get("mean_img")

    if vmap is None:
        return

    nz = vmap.shape[0]
    ncols = min(nz, 5)
    nrows = int(np.ceil(nz / ncols))

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


def save_segmentation_figures(job, fig_dir, stats=None):
    """Save cell detection diagnostic figures."""
    if stats is None:
        try:
            seg_results = job.load_segmentation_results()
        except Exception:
            print("  Could not load segmentation results for figures")
            return
        stats = seg_results.get("stats", seg_results) if isinstance(seg_results, dict) else seg_results

    if not hasattr(stats, '__len__') or len(stats) == 0:
        return

    res = job.load_corr_map_results()
    vmap = res.get("vmap")
    if vmap is None:
        return

    nz = vmap.shape[0]
    ncols = min(nz, 5)
    nrows = int(np.ceil(nz / ncols))

    cell_masks = np.zeros_like(vmap)
    for stat in stats:
        if "coords" in stat and "lam" in stat:
            cz, cy, cx = stat["coords"]
            lam = stat["lam"]
            if len(lam) > 0:
                cell_masks[cz, cy, cx] = np.maximum(cell_masks[cz, cy, cx], lam / lam.max())

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    for z in range(nz):
        ax = axes[z // ncols, z % ncols]
        vmax = np.percentile(vmap[vmap > 0], 99) if (vmap > 0).any() else 1
        ax.imshow(vmap[z], cmap="gray", vmin=0, vmax=vmax)
        mask_z = cell_masks[z]
        if mask_z.max() > 0:
            overlay = np.zeros((*mask_z.shape, 4))
            overlay[..., 0] = 1.0
            overlay[..., 3] = mask_z / mask_z.max() * 0.6
            ax.imshow(overlay)
        ax.set_title(f"Plane {z} ({(cell_masks[z] > 0).sum()} px)")
        ax.axis("off")
    for z in range(nz, nrows * ncols):
        axes[z // ncols, z % ncols].axis("off")
    fig.suptitle(f"Detected cells ({len(stats)} ROIs) overlaid on correlation map", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "07_cell_masks.png"), dpi=150)
    plt.close(fig)

    n_voxels = [len(s["lam"]) for s in stats if "lam" in s]
    if n_voxels:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(n_voxels, bins=50, edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Voxels per ROI")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"ROI size distribution (n={len(stats)})")

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
# Quality metrics visualization
# =============================================================================

def compute_and_plot_quality_metrics(job, stats, fig_dir, label="",
                                     min_npix_filter=10, mov=None):
    """Compute duplication/overmerge/shot noise metrics and save figures.

    Args:
        job: Job object (used for params like voxel_size_um, fs).
        stats: list of ROI stat dicts.
        fig_dir: directory to save figures.
        label: prefix for figure filenames.
        min_npix_filter: minimum voxels for an ROI to be included.
        mov: (nt, nz, ny, nx) movie array for overmerge computation (optional).

    Returns:
        dict of computed metrics.
    """
    from suite3d.quality_metrics import (
        compute_roi_metrics, duplicate_score_pairs, shot_noise_pct
    )

    voxel_size_um = job.params.get('voxel_size_um', (1, 1, 1))
    frate_hz = job.params.get('fs', 1.0)

    # Load F if available
    F = None
    try:
        roi_dir = job.dirs.get('rois', os.path.join(job.job_dir, 'rois'))
        F_path = os.path.join(roi_dir, 'F.npy')
        if os.path.exists(F_path):
            F = np.load(F_path)
    except Exception:
        pass

    metrics = compute_roi_metrics(
        stats, F=F, voxel_size_um=voxel_size_um, frate_hz=frate_hz,
        near_thresh=20.0, min_npix=min_npix_filter, mov=mov,
    )

    n_voxels = metrics['n_voxels']
    nc = metrics['n_rois']
    prefix = f"{label}_" if label else ""

    # --- Figure: size distribution with filter line ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.hist(n_voxels, bins=np.logspace(0, np.log10(max(n_voxels.max(), 1) + 1), 50),
            edgecolor="black", alpha=0.7)
    ax.axvline(min_npix_filter, color='red', ls='--', label=f'filter={min_npix_filter}')
    ax.set_xscale('log')
    ax.set_xlabel("Voxels per ROI")
    ax.set_ylabel("Count")
    n_above = (n_voxels >= min_npix_filter).sum()
    ax.set_title(f"Size distribution (n={nc}, {n_above} above filter)")
    ax.legend()

    # --- Duplication scatter ---
    ax = axes[1]
    dup_corrs = metrics['duplicate_corrs']
    dup_dists = metrics['duplicate_dists']
    if len(dup_corrs) > 0:
        ax.scatter(dup_dists, dup_corrs, s=1, alpha=0.3)
        ax.axhline(0.8, color='red', ls='--', alpha=0.7, label='dup threshold')
        n_dup = (dup_corrs > 0.8).sum()
        ax.set_title(f"Duplication: {n_dup} pairs > 0.8 corr")
    else:
        ax.set_title("Duplication: no F data")
    ax.set_xlabel("Distance (um)")
    ax.set_ylabel("Pairwise correlation")
    ax.legend()

    # --- Shot noise distribution ---
    ax = axes[2]
    shot_noise = metrics['shot_noise']
    valid_noise = shot_noise[~np.isnan(shot_noise)]
    if len(valid_noise) > 0:
        ax.hist(valid_noise, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Shot noise level")
        ax.set_ylabel("Count")
        ax.set_title(f"Shot noise (median={np.median(valid_noise):.3f})")
    else:
        ax.set_title("Shot noise: no F data")

    fig.suptitle(f"Quality Metrics {label}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{prefix}quality_metrics.png"), dpi=150)
    plt.close(fig)

    # --- Overmerge distribution ---
    om_scores = metrics['overmerge_scores']
    valid_om = om_scores[~np.isnan(om_scores)]
    if len(valid_om) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(valid_om, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(0.5, color='red', ls='--', label='overmerge threshold')
        n_overmerged = (valid_om > 0.5).sum()
        ax.set_xlabel("Overmerge score")
        ax.set_ylabel("Count")
        ax.set_title(f"Overmerge: {n_overmerged}/{len(valid_om)} above 0.5")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{prefix}overmerge.png"), dpi=150)
        plt.close(fig)

    print(f"  Saved quality metric figures to {fig_dir}")
    return metrics


# =============================================================================
# Parameter sweep
# =============================================================================

def run_segmentation_sweep(job, fig_dir):
    """Run a parameter sweep over key segmentation parameters.

    Sweeps vox_snr_thresh, peak_thresh, and roi_ext_iterations.
    For each combination, computes # ROIs, size distribution,
    duplication, and overmerge metrics.

    Returns:
        sweep_summary: dict from job.sweep_segmentation
        sweep_metrics: list of per-combination metric dicts
    """
    from suite3d.quality_metrics import compute_roi_metrics

    params_to_sweep = {
        'vox_snr_thresh': [0.02, 0.04, 0.08],
        'peak_thresh': [0.02, 0.03, 0.05],
    }

    # Ensure current values are in sweep
    for k, vals in params_to_sweep.items():
        if job.params[k] not in vals:
            vals.append(job.params[k])
            vals.sort()

    print(f"\n=== Segmentation sweep: {len(params_to_sweep)} params ===")
    for k, v in params_to_sweep.items():
        print(f"  {k}: {v}")

    sweep_summary = job.sweep_segmentation(
        params_to_sweep,
        sweep_name="seg-sweep",
        all_combinations=True,
        extract=True,
    )

    # Compute metrics for each combination
    voxel_size_um = job.params.get('voxel_size_um', (1, 1, 1))
    frate_hz = job.params.get('fs', 1.0)
    combinations = sweep_summary['combinations']
    param_names = sweep_summary['param_names']

    sweep_metrics = []
    for i, res in enumerate(sweep_summary['results']):
        stats = res.get('stats', [])
        if isinstance(stats, dict):
            stats = stats.get('stats', [])

        # Try to load F
        F = None
        roi_dir = res.get('roi_dir', '')
        if roi_dir:
            F_path = os.path.join(roi_dir, 'F.npy')
            if os.path.exists(F_path):
                F = np.load(F_path)

        m = compute_roi_metrics(
            stats, F=F, voxel_size_um=voxel_size_um, frate_hz=frate_hz,
            near_thresh=20.0, min_npix=10,
        )
        m['combination'] = {param_names[j]: combinations[i][j] for j in range(len(param_names))}
        sweep_metrics.append(m)

    # --- Sweep summary figure ---
    n_combs = len(sweep_metrics)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. # ROIs vs combination
    n_rois_arr = [m['n_rois'] for m in sweep_metrics]
    n_above_filter = [(m['n_voxels'] >= 10).sum() for m in sweep_metrics]
    labels = ["\n".join(f"{k}={v}" for k, v in m['combination'].items()) for m in sweep_metrics]

    ax = axes[0, 0]
    x = np.arange(n_combs)
    ax.bar(x - 0.15, n_rois_arr, 0.3, label='Total ROIs', alpha=0.7)
    ax.bar(x + 0.15, n_above_filter, 0.3, label='ROIs >= 10 vox', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("# ROIs")
    ax.set_title("ROI count per sweep combination")
    ax.legend()

    # 2. Size distribution (box plot)
    ax = axes[0, 1]
    size_data = [m['n_voxels'][m['n_voxels'] >= 10] for m in sweep_metrics]
    bp = ax.boxplot(size_data, showfliers=False)
    ax.set_xticklabels([f"C{i}" for i in range(n_combs)], fontsize=8)
    ax.set_ylabel("Voxels per ROI")
    ax.set_title("ROI size distribution (>= 10 vox)")
    ax.set_yscale('log')

    # 3. Duplication index
    ax = axes[1, 0]
    dup_counts = []
    for m in sweep_metrics:
        if len(m['duplicate_corrs']) > 0:
            dup_counts.append((m['duplicate_corrs'] > 0.8).sum())
        else:
            dup_counts.append(0)
    ax.bar(x, dup_counts, alpha=0.7, color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("# duplicate pairs (corr > 0.8)")
    ax.set_title("Duplication index per combination")

    # 4. Shot noise distribution
    ax = axes[1, 1]
    noise_medians = []
    for m in sweep_metrics:
        valid = m['shot_noise'][~np.isnan(m['shot_noise'])]
        noise_medians.append(np.median(valid) if len(valid) > 0 else 0)
    ax.bar(x, noise_medians, alpha=0.7, color='green')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("Median shot noise")
    ax.set_title("Shot noise per combination")

    fig.suptitle("Segmentation Parameter Sweep Summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "sweep_summary.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved sweep summary to {fig_dir}")
    return sweep_summary, sweep_metrics


# =============================================================================
# Pipeline
# =============================================================================

def run_pipeline(data_dir, output_dir, job_id, skip_init, skip_register,
                 skip_corrmap, skip_segment, skip_extract, export_dir,
                 do_sweep=False, compute_quality=True):
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
        tic = time.time()
        job.calculate_corr_map()
        print(f"  Corrmap took {(time.time() - tic)/60:.1f} min")
        save_corrmap_figures(job, fig_dir)

    if not skip_segment:
        print("\n=== Segmenting ROIs ===")
        tic = time.time()
        job.segment_rois()
        print(f"  Segmentation took {(time.time() - tic)/60:.1f} min")
        save_segmentation_figures(job, fig_dir)

    if not skip_extract:
        print("\n=== Computing neuropil masks and extracting traces ===")
        job.compute_npil_masks()
        job.extract_and_deconvolve()

    if compute_quality and not do_sweep:
        print("\n=== Computing quality metrics ===")
        try:
            seg_results = job.load_segmentation_results()
            stats = seg_results.get("stats", seg_results) if isinstance(seg_results, dict) else seg_results
            compute_and_plot_quality_metrics(job, stats, fig_dir, label="base")
        except Exception as e:
            print(f"  Quality metrics failed: {e}")

    if do_sweep:
        print("\n=== Running segmentation parameter sweep ===")
        run_segmentation_sweep(job, fig_dir)

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
    parser.add_argument("--sweep", action="store_true",
                        help="Run segmentation parameter sweep after pipeline")
    parser.add_argument("--no_quality", action="store_true",
                        help="Skip quality metric computation")

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
        do_sweep=args.sweep,
        compute_quality=not args.no_quality,
    )


if __name__ == "__main__":
    main()
