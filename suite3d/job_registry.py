"""
Lightweight central registry of suite3d jobs.

Stores a JSON file at ~/.suite3d/jobs.json so that UIs (napari viewer, web app)
can list known jobs without scanning the filesystem.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

SUITE3D_DIR = Path.home() / ".suite3d"
REGISTRY_PATH = SUITE3D_DIR / "jobs.json"


def _ensure_registry():
    """Create ~/.suite3d/ and jobs.json if they don't exist."""
    SUITE3D_DIR.mkdir(exist_ok=True)
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.write_text("[]")


def _load_registry():
    """Return the list of job entries from disk."""
    _ensure_registry()
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def _save_registry(entries):
    """Write the list of job entries to disk."""
    _ensure_registry()
    with open(REGISTRY_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def register_job(job_dir, job_id=None):
    """Register a job in the central registry.

    If the job path is already registered, updates its metadata.
    Called automatically by Job.init_job_dir().

    Args:
        job_dir: Absolute path to the s3d-{job_id} directory.
        job_id: Optional job ID string. Inferred from directory name if not provided.
    """
    job_dir = str(Path(job_dir).resolve())

    if job_id is None:
        dirname = os.path.basename(job_dir)
        job_id = dirname[4:] if dirname.startswith("s3d-") else dirname

    now = datetime.now(timezone.utc).isoformat()
    entries = _load_registry()

    for entry in entries:
        if entry["path"] == job_dir:
            entry["job_id"] = job_id
            entry["last_opened"] = now
            _save_registry(entries)
            return

    entries.append({
        "job_id": job_id,
        "path": job_dir,
        "created": now,
        "last_opened": now,
    })
    _save_registry(entries)


def touch_job(job_dir):
    """Update last_opened timestamp for an existing job entry."""
    job_dir = str(Path(job_dir).resolve())
    entries = _load_registry()
    for entry in entries:
        if entry["path"] == job_dir:
            entry["last_opened"] = datetime.now(timezone.utc).isoformat()
            _save_registry(entries)
            return


def get_registered_jobs():
    """Return all registered jobs, sorted by last_opened (most recent first).

    Returns:
        List of dicts with keys: job_id, path, created, last_opened
    """
    entries = _load_registry()
    # Filter out entries whose directories no longer exist
    valid = [e for e in entries if os.path.isdir(e["path"])]
    if len(valid) != len(entries):
        _save_registry(valid)
    return sorted(valid, key=lambda e: e.get("last_opened", ""), reverse=True)


def remove_job(job_dir):
    """Remove a job from the registry (does not delete files)."""
    job_dir = str(Path(job_dir).resolve())
    entries = _load_registry()
    entries = [e for e in entries if e["path"] != job_dir]
    _save_registry(entries)


def scan_job_dir(job_dir):
    """Scan a directory and return a manifest of available outputs.

    Handles three cases:
    - A job root (s3d-{id}/) containing rois/, corrmap/, sweeps/ subdirs
    - A leaf output dir (e.g. a rois/ dir with stats.npy directly in it)
    - A sweep dir (with sweep_summary.npy directly in it)

    Args:
        job_dir: Path to scan.

    Returns:
        Dict describing what's available, e.g.:
        {
            "rois": "/path/to/rois",
            "corrmap": "/path/to/corrmap",
            "sweeps": {"sweep_name": "/path/to/sweeps/sweep_name", ...}
        }
    """
    job_dir = Path(job_dir)
    manifest = {}

    # Case 1: this directory itself contains stats.npy (user pointed at a rois/ dir)
    if (job_dir / "stats.npy").exists() or (job_dir / "stats_small.npy").exists():
        manifest["rois"] = str(job_dir)
        return manifest

    # Case 2: this directory itself contains sweep_summary.npy
    if (job_dir / "sweep_summary.npy").exists():
        manifest["sweeps"] = {job_dir.name: str(job_dir)}
        return manifest

    # Case 3: this directory itself contains vmap.npy (user pointed at corrmap/ dir)
    if (job_dir / "vmap.npy").exists():
        manifest["corrmap"] = str(job_dir)
        return manifest

    # Case 4: job root — look for subdirectories
    rois_dir = job_dir / "rois"
    if rois_dir.exists() and (
        (rois_dir / "stats.npy").exists() or (rois_dir / "stats_small.npy").exists()
    ):
        manifest["rois"] = str(rois_dir)

    corrmap_dir = job_dir / "corrmap"
    if corrmap_dir.exists() and (corrmap_dir / "vmap.npy").exists():
        manifest["corrmap"] = str(corrmap_dir)

    sweeps_dir = job_dir / "sweeps"
    if sweeps_dir.exists():
        sweeps = {}
        for child in sorted(sweeps_dir.iterdir()):
            if child.is_dir() and (child / "sweep_summary.npy").exists():
                sweeps[child.name] = str(child)
        if sweeps:
            manifest["sweeps"] = sweeps

    return manifest
