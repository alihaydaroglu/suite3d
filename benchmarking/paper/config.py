"""Locked configuration for the speed comparison.

One dataset, one parameter set per tool. Paths assume container mount
layout (`/data` for inputs, `/results` for outputs) but can be
overridden via CLI flags on the runners.
"""
from pathlib import Path

DATASET = "TC030_2025-03-25"

# Container mount points (host paths get bound to these via -v).
DATA_ROOT = Path("/data")
RESULTS_ROOT = Path("/results")

# Within DATA_ROOT, where the raw TIFs live. For the speed run we use
# TC030 / 2025-03-25 / expnum 3 only (mirrors host
# /mnt/zenneth-subjects/TC030/2025-03-25/3/).
RAW_TIF_DIRS = [
    DATA_ROOT / "raw" / "TC030" / "2025-03-25" / "3",
]
S3D_JOB_BASE = DATA_ROOT / "suite3d"
S3D_JOB_ID = DATASET

# -- Imaging geometry (matches fig-comparison-contained config) --
N_PLANES_TOTAL = 9
FUNCTIONAL_PLANES = [2, 3, 4, 5, 6, 7, 8]
NUM_COLORS = 2
FUNCTIONAL_COLOR_CHANNEL = 0
GREEN_CHANNEL_IDX = 1
TAU = 1.3
VOXEL_SIZE_UM = (10, 1, 1)

# -- Suite3D job params (canonical TC030 set) --
S3D_JOB_PARAMS = dict(
    tau=TAU,
    lbm=False,
    num_colors=NUM_COLORS,
    functional_color_channel=FUNCTIONAL_COLOR_CHANNEL,
    voxel_size_um=VOXEL_SIZE_UM,
    n_init_files=4,
    nonrigid=True,
    subtract_crosstalk=False,
    fuse_strips=False,
    max_shift_nr=(1, 5, 5),
    nr_npad=(1, 3, 3),
    block_size_3d=(4, 128, 128),
)
S3D_JOB_PARAMS["3d_reg"] = True
S3D_JOB_PARAMS["gpu_reg"] = True

# -- Suite3D segmentation params (canonical set) --
S3D_SEG_PARAMS = dict(
    extend_thresh=0.04,
    peak_thresh=0.03,
    activity_thresh=5.0,
    roi_ext_iterations=20,
    roi_dilations_per_iter=2,
    max_pix=10000,
    segmentation_timebin=2,
    use_power_iter_v1=True,
    multi_source=True,
    ext_subtract_iters=2,
    patch_size_xy=(300, 300),
    patch_overlap_xy=(75, 75),
    min_frames=100,
    segmentation_spatial_filt=1,
    n_proc_detect=32,
    vox_snr_thresh=0.05,
)

S3D_CORRMAP_PARAMS = dict(
    cell_filt_xy_um=1.5,
    cell_filt_z_um=10,
    intensity_thresh=5,
    detection_timebin=3,
)

# -- CaImAn params (provisional middle-of-grid pick) --
CAIMAN_PARAMS = dict(
    K=1000,
    merge_thresh=0.8,
    rval_thr=0.9,
    gSig=(5, 5, 2),
    p=2,
    fr=4,
    decay_time=1.0,
    min_SNR=3,
    use_cnn=False,
    # motion correction
    strides=(24, 24, 10),
    overlaps=(12, 12, 2),
    max_shifts=(4, 4, 2),
    max_deviation_rigid=5,
    pw_rigid=True,
    # online (OnACID)
    init_method="cnmf",
    init_batch=200,
    epochs=1,
    n_refit=0,
    is3D=True,
)
