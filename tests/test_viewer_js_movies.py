"""The JS must ask for exactly the movie files Python wrote.

`frameURL` in assets/viewer.js and `frame_filename` in viewer/movies.py encode the
same naming convention in two languages. If they drift, every frame 404s and the
movie panel silently shows nothing -- there is no error to catch on file://, an
<img> that fails to load just stays blank. So run the shipped JS under node.

Skipped when node is unavailable.
"""
import json
import os
import shutil
import subprocess
import textwrap

import pytest

from suite3d.viewer import curation as cur
from suite3d.viewer.movies import frame_filename

NODE = shutil.which("node")
ASSETS = os.path.join(os.path.dirname(cur.__file__), "assets")

pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


def _extract(name):
    src = open(os.path.join(ASSETS, "viewer.js")).read()
    start = src.index(f"function {name}(")
    return src[start:src.index("\n}", start) + 2]


def _run(body):
    out = subprocess.run([NODE, "-e", textwrap.dedent(body)],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


EXT = {"registered": "jpg", "mov_sub": "png"}   # ext is per movie, not per frame
CASES = [("registered", 0, 0), ("registered", 21, 1234), ("mov_sub", 7, 59)]


def test_frame_urls_match_the_files_python_writes():
    movies = {k: {"ext": e} for k, e in EXT.items()}
    js = _run(f"""
        const MOVIES = {json.dumps(movies)};
        {_extract("frameURL")}
        console.log(JSON.stringify({json.dumps(CASES)}
            .map(([k, z, t]) => frameURL(k, z, t))));
    """)
    got = json.loads(js)
    want = [f"viewer/movies/{k}/{frame_filename(z, t, EXT[k])}" for k, z, t in CASES]
    assert got == want


def test_trace_cursor_index_matches_the_meta_written_by_python():
    """frame -> sample on the trace time axis, using meta's t0 + stride."""
    movies = {"mov_sub": {"t0": 10, "stride": 5}}
    js = _run(f"""
        const MOVIES = {json.dumps(movies)};
        const state = {{movie: "mov_sub", frame: 4}};
        {_extract("traceIndexOfFrame")}
        console.log(traceIndexOfFrame());
    """)
    assert int(js) == 10 + 4 * 5

    js = _run(f"""
        const MOVIES = {json.dumps(movies)};
        const state = {{movie: null, frame: 4}};
        {_extract("traceIndexOfFrame")}
        console.log(traceIndexOfFrame());
    """)
    assert int(js) == -1                    # no movie -> no cursor
