#!/usr/bin/env python
"""Make an executed notebook safe to publish.

The demo walkthroughs are rendered to static HTML for the public website, so
their *output cells* must not carry a filesystem layout, a username, or a
machine name. Suite3D's job logger prints absolute tif paths, so executing a
notebook against real data always leaks some.

This does three things, in order:

1. Rewrites absolute paths in output cells to neutral placeholders.
2. Removes `ipywidgets` output (nbconvert renders a stateless widget as a
   silently empty div) and any widget state left in notebook metadata.
3. **Verifies** nothing leaked, and exits non-zero if it did. That check is the
   point — the rewriting is best-effort, the assertion is not.

    python tools/scrub_outputs.py 01-v1-tc030/walkthrough.ipynb
    python tools/scrub_outputs.py --check-only */walkthrough.ipynb
"""

import argparse
import json
import re
import sys
from pathlib import Path


# Ordered: the most specific prefix must win, so a figshare path becomes
# <data-root> rather than <path>/suite3d_figshare_data/...
REWRITES = [
    (re.compile(r"/mnt/znas-share/suite3d_figshare_data/?"), "<data-root>/"),
    (re.compile(r"/mnt/md0/s3d-figshare-staging/?"),         "<data-root>/"),
    (re.compile(r"/mnt/md0/s3d-revisions/suite3d/?"),        "<results>/"),
    (re.compile(r"/home/[a-z0-9_-]+/"),                      "<home>/"),
    (re.compile(r"/mnt/[a-z0-9_.-]+/"),                      "<path>/"),
]

# Anything matching these in an output cell is a hard failure.
FORBIDDEN = re.compile(r"/mnt/|/home/[a-z0-9_-]+|/Users/[A-Za-z0-9_-]+")


def scrub_text(s):
    for pat, repl in REWRITES:
        s = pat.sub(repl, s)
    return s


def scrub_output(out):
    """Rewrite text payloads of one output; return True if it should be dropped."""
    if out.get("output_type") == "display_data":
        data = out.get("data", {})
        if any(k.startswith("application/vnd.jupyter.widget") for k in data):
            return True  # drop: renders as an empty div

    for key in ("text",):
        if key in out:
            out[key] = [scrub_text(x) for x in out[key]]

    for key in ("data",):
        if key in out:
            for mime, val in list(out[key].items()):
                if mime.startswith("image/"):
                    continue  # binary payload; nothing to scrub
                if isinstance(val, list):
                    out[key][mime] = [scrub_text(x) for x in val]
                elif isinstance(val, str):
                    out[key][mime] = scrub_text(val)

    if out.get("output_type") == "error":
        out["evalue"] = scrub_text(out.get("evalue", ""))
        out["traceback"] = [scrub_text(x) for x in out.get("traceback", [])]

    return False


def _strings(val):
    """Flatten a notebook payload to the strings inside it.

    Output payloads are not uniformly text: `text/plain` is a list of str,
    but a widget's `application/vnd.jupyter.widget-view+json` is a dict, and
    nesting is allowed. Walk it rather than assuming.
    """
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        return [s for v in val for s in _strings(v)]
    if isinstance(val, dict):
        return [s for v in val.values() for s in _strings(v)]
    return []


def collect_output_text(nb):
    """Every human-readable string in every output cell."""
    chunks = []
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            chunks.extend(_strings(out.get("text", [])))
            for mime, val in out.get("data", {}).items():
                if mime.startswith("image/"):
                    continue  # binary payload
                chunks.extend(_strings(val))
            if out.get("output_type") == "error":
                chunks.extend(_strings(out.get("evalue", "")))
                chunks.extend(_strings(out.get("traceback", [])))
    return chunks


def process(path, check_only=False):
    nb = json.loads(path.read_text())

    n_exec = sum(1 for c in nb["cells"]
                 if c["cell_type"] == "code" and c.get("outputs"))
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")

    if not check_only:
        for cell in nb["cells"]:
            if "outputs" in cell:
                cell["outputs"] = [o for o in cell["outputs"] if not scrub_output(o)]
        nb.get("metadata", {}).pop("widgets", None)
        path.write_text(json.dumps(nb, indent=1) + "\n")

    leaks = [c for c in collect_output_text(nb) if FORBIDDEN.search(c)]
    status = "OK " if not leaks else "LEAK"
    print(f"  [{status}] {path}  ({n_exec}/{n_code} code cells have outputs)")
    for c in leaks[:5]:
        m = FORBIDDEN.search(c)
        print(f"         -> {c[max(0, m.start()-20):m.end()+40].strip()!r}")
    if n_exec == 0:
        print("         note: no executed outputs — nothing to publish yet")
    return len(leaks)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebooks", nargs="+", type=Path)
    ap.add_argument("--check-only", action="store_true",
                    help="verify without rewriting (use in CI)")
    args = ap.parse_args()

    total = sum(process(p, args.check_only) for p in args.notebooks)
    if total:
        print(f"\nFAIL: {total} absolute path(s) remain in output cells.", file=sys.stderr)
        return 1
    print("\nAll clean — safe to publish.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
