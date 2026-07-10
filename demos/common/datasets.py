"""Per-dataset recipes for the Suite3D demos.

Each entry holds only what the *data* forces: acquisition geometry, plus the
handful of parameters a modality genuinely needs. Everything else is inherited
from ``suite3d.default_params``, so a demo shows you the shipped defaults rather
than a wall of overrides.

Two rules this file exists to enforce:

1. ``fs`` is the **volume** rate (``SI.hRoiManager.scanVolumeRate``), hard-coded
   per dataset. Do **not** call ``suite3d.io.get_vol_rate()`` — despite its name
   it returns the per-plane frame rate. Suite3D derives
   ``detection_timebin = 2 * round(fs / tau)``, so a wrong ``fs`` silently
   changes the correlation map, not merely a plot's time axis.

2. ``voxel_size_um`` is ``(dz, dy, dx)`` and is not cosmetic: the correlation
   map divides ``cell_filt_z_um`` and ``npil_filt_z_um`` by ``dz``.

Values are those of the published reference runs. See the figshare archive's
``reproduce_segmentation.py`` for the paper-exact recipes.
"""

import numpy as np


# ``multi_source`` shipped as True until 2026-07-09 and now defaults to False.
# The three reference runs were all produced with it ON, so the demos pin it
# explicitly rather than silently tracking the default.
MULTI_SOURCE = True


DATASETS = {
    "v1": {
        "label": "V1 — standard multi-plane 2P, mouse cortex (TC030)",
        "folder": "v1",
        "job_id": "demo-v1",
        "expected_rois": 845,
        "params": {
            "fs": 3.335425084175085,      # volume rate (Hz)
            "tau": 1.3,                   # GCaMP6s decay (s)
            "voxel_size_um": (10, 1, 1),  # (dz, dy, dx) microns
            "planes": np.array([2, 3, 4, 5, 6, 7, 8]),
            "n_ch_tif": 9,                # planes per volume in the tif
            "num_colors": 2,
            "functional_color_channel": 0,
            "lbm": False,
            "faced": False,
            "subtract_crosstalk": False,
            "fuse_strips": False,
            "3d_reg": True,
            "apply_z_shift": True,
            "multi_source": MULTI_SOURCE,
        },
    },

    "lbm": {
        "label": "LBM — light-beads microscopy, ~40k neurons (SS004)",
        "folder": "lbm",
        "job_id": "demo-lbm",
        # The reference run (`s3d-ss004-5min-default`, 1300 volumes) found
        # 43,652 ROIs. Every detection and segmentation parameter below matches
        # it, but registration does not (see `nonrigid`), so treat this as a
        # sanity check rather than an exact target.
        "expected_rois": None,
        "params": {
            "fs": 4.116358658453114,
            "tau": 1.3,
            "voxel_size_um": (20, 3.33, 3.33),
            # ScanImage records the cavities interleaved; this reorders them
            # into depth order. Specific to the microscope.
            "planes": np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                                1, 3, 5, 7, 9, 11, 13, 15, 17]),
            "n_ch_tif": 26,
            "num_colors": 1,
            "functional_color_channel": 0,
            "lbm": True,
            "faced": False,
            # --- LBM-specific acquisition handling
            "cavity_size": 13,           # planes per cavity
            "subtract_crosstalk": True,  # between cavities
            "fuse_strips": True,         # mesoscope strips
            "fuse_shift_override": 6,
            "3d_reg": True,
            "apply_z_shift": False,

            # The init pass estimates the cavity crosstalk coefficient, and that
            # estimate is SUBTRACTED from the movie (subtract_crosstalk=True).
            # It is noisy from a single tif: with the shipped default of 1 file
            # it lands at 0.080, against the reference run's 0.160 -- the same
            # per-plane profile, scaled ~2x down. The reference used 4.
            "n_init_files": 4,

            # Registration. The published 43,652-ROI run used rigid-only
            # registration; this demo deliberately uses nonrigid (@ali), which
            # is the better default. Registration feeds the correlation map, so
            # the ROI count here will be close to, but not exactly, 43,652.
            "nonrigid": True,

            # Post-extraction trace deduplication. Measured on the reference
            # job: at the shipped thresholds it removes 0 of 43,652 ROIs.
            # `deduplication_thresh_um=15.0` is smaller than this recording's
            # 20 um z-spacing, so it cannot merge a cell split across adjacent
            # planes -- which is the only duplicate LBM really produces.
            "deduplicate": True,

            # --- Detection (correlation map) ---------------------------------
            # Pinned to the values that produced the published 43,652-ROI
            # segmentation of this recording (`s3d-ss004-5min-default`). Several
            # of these happen to equal today's shipped defaults, and are pinned
            # anyway: a future change to `default_params.py` must not silently
            # move this demo's ROI count. (The `multi_source` default already
            # flipped once, on 2026-07-09.)
            #
            # LBM has low per-voxel SNR, hence a gentler intensity gate and a
            # coarser detection time bin than a conventional 2P recording.
            "intensity_thresh": 1.0,
            "detection_timebin": 6,
            "cell_filt_xy_um": 1.5,
            "cell_filt_z_um": 10,
            "npil_filt_xy_um": 100.0,
            "npil_filt_z_um": 15.0,

            # --- Segmentation ------------------------------------------------
            "peak_thresh": 0.1,
            "vox_snr_thresh": 0.05,
            "ext_subtract_iters": 0,
            "roi_dilations_per_iter": 2,
            "segmentation_spatial_filt": 2,
            "segmentation_timebin": 1,
            "multi_source": MULTI_SOURCE,

            "split_tif_size": 100,
        },
    },

    "hippocampus": {
        "label": "Hippocampus — standard 2P, densely packed CA1 (ATL020)",
        "folder": "hippocampus",
        "job_id": "demo-hippocampus",
        "expected_rois": 1347,
        "params": {
            "fs": 6.00468,
            "tau": 1.3,
            "voxel_size_um": (10, 1, 1),
            "planes": np.array([1, 2, 3, 4]),
            "n_ch_tif": 5,
            "num_colors": 2,
            "functional_color_channel": 0,
            "lbm": False,
            "faced": False,
            "subtract_crosstalk": False,
            "fuse_strips": False,
            "3d_reg": True,
            # 6.6% of volumes peg at the +-2-plane axial search cap on this
            # 4-plane stack, so applying the estimate would translate those
            # frames by 62% of the volume. Measure z in 3D, but do not apply it.
            "apply_z_shift": False,
            "multi_source": MULTI_SOURCE,
        },
    },
}


def get_params(name):
    """Return a fresh copy of the parameter dict for one dataset."""
    if name not in DATASETS:
        raise KeyError(
            "Unknown dataset %r. Known: %s" % (name, ", ".join(sorted(DATASETS)))
        )
    params = dict(DATASETS[name]["params"])
    check_volume_rate(params["fs"], len(params["planes"]), name)
    return params


def check_volume_rate(fs, n_planes, name=""):
    """Guard against `fs` having been set to the per-plane frame rate.

    A volume rate above ~15 Hz on a multi-plane stack is almost always the
    plane rate that `io.get_vol_rate()` returns. Raising here is much kinder
    than a quietly wrong correlation map.
    """
    if n_planes > 1 and fs > 15.0:
        raise ValueError(
            "%s: fs=%.4f Hz on a %d-plane volume looks like the per-plane "
            "frame rate, not the volume rate (implied volume rate %.4f Hz). "
            "Suite3D derives detection_timebin from fs, so this would corrupt "
            "the correlation map. Use SI.hRoiManager.scanVolumeRate; note that "
            "suite3d.io.get_vol_rate() returns the PLANE rate."
            % (name or "dataset", fs, n_planes, fs / n_planes)
        )
    return fs
