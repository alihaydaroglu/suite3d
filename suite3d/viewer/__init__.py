"""Portable, offline HTML viewer for Suite3D segmentation results.

    job.make_html_viewer()   -> s3d-results-<job_id>/viewer.html

The output directory is self-contained: copy it to another machine and
double-click `viewer.html`.  No server, no Python, no internet.
"""

from .curation import apply_curation, dump_curation, load_curation, stat_table
from .html_viewer import import_curation, make_html_viewer
from .movies import DEFAULT_SPEC, normalize_spec

__all__ = [
    "make_html_viewer", "import_curation",
    "apply_curation", "stat_table", "load_curation", "dump_curation",
    "normalize_spec", "DEFAULT_SPEC",
]
