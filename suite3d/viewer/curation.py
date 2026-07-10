"""`iscell` as a derived value.

The napari curation UI stores `iscell` as *the* state: filter sliders write into
it, manual clicks write into it, and the two race -- a slider drag silently
undoes hand curation, and "reset filters" destroys it.

Here `iscell` is never stored.  It is derived from two small objects:

    filters : {stat_name: [lo, hi]}   slider positions (inclusive bounds)
    manual  : {roi_id: 0 | 1}         sparse explicit overrides

    iscell(roi) = manual[roi]                      if roi in manual
                = all(lo <= stat[k][roi] <= hi)    otherwise

A few kB, not 43k booleans.  A slider can never clobber a manual decision; each
manual click is one undo entry; resetting the filters keeps the hand curation; and
the entire session is reproducible from a tiny JSON.

THIS PREDICATE IS DUPLICATED IN JS (assets/viewer.js, `applyCuration`).  The two
must agree exactly -- `test_curation_parity` in the test-suite compares them, and
`Job.import_curation()` is meaningless if they drift.
"""

import json

import numpy as np

SCHEMA_VERSION = 1


def stat_table(stats):
    """The per-ROI scalars the filter sliders operate on.

    Kept small and derived only from `stats.npy`, so the same table is available in
    the browser (it ships in meta.js) and in Python.
    """
    npix = np.array([len(s["lam"]) for s in stats], dtype=np.float32)
    zspan = np.array([len(np.unique(np.asarray(s["coords"])[0])) for s in stats],
                     dtype=np.float32)
    peak = np.array([float(s.get("peak_val", np.nan)) for s in stats], dtype=np.float32)
    snr = np.array([float(np.median(s["vox_snrs"])) if "vox_snrs" in s else np.nan
                    for s in stats], dtype=np.float32)
    return {"npix": npix, "zspan": zspan, "peak_val": peak, "vox_snr": snr}


def apply_curation(table, filters=None, manual=None, n_rois=None):
    """Derive the boolean `iscell` array.  See module docstring for the rule.

    NaN stats never fail a filter -- a missing statistic must not silently delete
    ROIs.  (`peak_val`/`vox_snr` are absent from some legacy stats.npy.)
    """
    if n_rois is None:
        n_rois = len(next(iter(table.values())))
    keep = np.ones(n_rois, dtype=bool)
    for name, (lo, hi) in (filters or {}).items():
        if name not in table:
            raise KeyError(f"unknown filter stat {name!r}; have {sorted(table)}")
        v = table[name]
        ok = ~np.isfinite(v) | ((v >= lo) & (v <= hi))
        keep &= ok
    for roi, val in (manual or {}).items():
        keep[int(roi)] = bool(int(val))
    return keep


def load_curation(path):
    with open(path) as f:
        d = json.load(f)
    if d.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"curation schema {d.get('schema')} != {SCHEMA_VERSION}")
    return d


def dump_curation(path, filters, manual, job_id, n_rois):
    with open(path, "w") as f:
        json.dump(dict(schema=SCHEMA_VERSION, job_id=job_id, n_rois=int(n_rois),
                       filters=filters, manual={str(k): int(v) for k, v in manual.items()}),
                  f, indent=1)
