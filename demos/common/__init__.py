"""Shared helpers for the Suite3D demos.

Each demo subdirectory is standalone; they import this package via a small
`sys.path` shim at the top of each script so that `python 01-v1-tc030/
run_pipeline.py` works from anywhere without installing the demos.
"""

from .datasets import DATASETS, get_params, check_volume_rate
from .pipeline import (
    build_parser,
    extract_batch,
    find_tifs,
    load_or_create_job,
    log,
    main,
    n_rois,
    open_viewer,
    run_stages,
)

__all__ = [
    "DATASETS", "get_params", "check_volume_rate",
    "build_parser", "extract_batch", "find_tifs", "load_or_create_job", "log", "main",
    "n_rois", "open_viewer", "run_stages",
]
