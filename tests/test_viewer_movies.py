"""Movie-snippet writer: axis order, pixel budget, and what lands on disk."""
import os

import numpy as np
import pytest

from suite3d.viewer import movies as MV


class _FakeJob:
    """Only the surface `write_movie_snippets` touches."""

    def __init__(self, tmp, reg=None, sub=None, params=None):
        self.dirs = {"job_dir": str(tmp)}
        self.params = params or {}
        self._reg, self._sub = reg, sub
        self.lines = []
        if sub is not None:
            os.makedirs(os.path.join(tmp, "mov_sub"), exist_ok=True)

    def log(self, msg, level=0):
        self.lines.append(msg)

    def get_registered_movie(self):
        return self._reg

    def get_subtracted_movie(self):
        return self._sub


def test_normalize_spec_accepts_true_int_and_dict():
    assert MV.normalize_spec(True) == MV.DEFAULT_SPEC
    assert MV.normalize_spec(30)["n_frames"] == 30
    assert MV.normalize_spec({"quality": 60})["quality"] == 60
    with pytest.raises(ValueError, match="unknown movie_snippet keys"):
        MV.normalize_spec({"n_frame": 10})          # typo must not be silently ignored
    with pytest.raises(TypeError):
        MV.normalize_spec("100")


def test_budget_downsamples_before_dropping_frames():
    spec = dict(MV.DEFAULT_SPEC, n_frames=100, max_pixels=1e7)
    ds, nf = MV._budget(spec, 2, 4, 500, 500)       # 2e8 px at ds=1
    assert ds > 1 and nf == 100, (ds, nf)           # a coarse movie beats a short one
    # ds is capped, so an impossible budget must trim frames instead
    ds, nf = MV._budget(dict(spec, max_pixels=1e4), 2, 4, 500, 500)
    assert ds == 8 and nf < 100

    ds, nf = MV._budget(dict(spec, downsample=3, max_pixels=1e12), 2, 4, 500, 500)
    assert (ds, nf) == (3, 100)                     # explicit downsample is honoured


def test_resolve_planes_defaults_to_all_and_validates():
    assert MV.resolve_planes(MV.normalize_spec(True), 4) == [0, 1, 2, 3]
    assert MV.resolve_planes(MV.normalize_spec({"planes": [3, 1, 1]}), 4) == [1, 3]
    with pytest.raises(ValueError, match="out of range"):
        MV.resolve_planes(MV.normalize_spec({"planes": [0, 9]}), 4)
    with pytest.raises(ValueError, match="non-empty"):
        MV.normalize_spec({"planes": []})


def test_planes_subset_writes_only_those_planes_and_records_them(tmp_path):
    reg, sub = _movies(nz=5, nt=20)
    job = _FakeJob(tmp_path, reg, sub)
    vdir = str(tmp_path / "v")
    meta = MV.write_movie_snippets(job, vdir, MV.normalize_spec({"n_frames": 2, "planes": [0, 3]}),
                                   nz=5, ny=8, nx=8)
    for key in meta:
        assert meta[key]["planes"] == [0, 3]
        got = sorted(f[:3] for f in os.listdir(os.path.join(vdir, "movies", key)))
        assert set(got) == {"z00", "z03"}, got


def test_planes_subset_relaxes_the_pixel_budget(tmp_path):
    """Budget counts written planes, not the whole volume -- else 3 of 22 planes
    would still be downsampled as if all 22 were being shipped."""
    spec = dict(MV.DEFAULT_SPEC, n_frames=60, max_pixels=6e8)
    ds_all, _ = MV._budget(spec, 2, 22, 928, 735)
    ds_few, _ = MV._budget(spec, 2, 3, 928, 735)
    assert ds_all > 1 and ds_few == 1


def test_downsample_is_a_block_mean():
    a = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    d = MV._downsample(a, 2)
    assert d.shape == (2, 2, 2)
    assert d[0, 0, 0] == pytest.approx((0 + 1 + 4 + 5) / 4)
    a5 = np.ones((1, 5, 5), np.float32)             # ragged edge is cropped, not padded
    assert MV._downsample(a5, 2).shape == (1, 2, 2)


def test_plane_frames_axis_order_differs_between_movies():
    """The one mistake that silently yields a plausible movie of the wrong plane."""
    nz, nt, ny, nx = 3, 6, 4, 5
    reg = np.arange(nz * nt * ny * nx, dtype=np.float32).reshape(nz, nt, ny, nx)
    sub = np.moveaxis(reg, 0, 1).copy()             # same data, (nt, nz, ny, nx)

    got_reg = MV._plane_frames(reg, MV.REGISTERED, z=2, t0=1, n_frames=3)
    got_sub = MV._plane_frames(sub, MV.DETECTION, z=2, t0=1, n_frames=3)
    assert got_reg.shape == (3, ny, nx)
    assert np.array_equal(got_reg, reg[2, 1:4])
    assert np.array_equal(got_sub, got_reg)         # identical data => identical frames


def _movies(nz=2, nt=20, ny=8, nx=8, timebin=1, rng=None):
    rng = rng or np.random.default_rng(0)
    reg = rng.random((nz, nt, ny, nx)).astype(np.float32)
    sub = rng.random((nt // timebin, nz, ny, nx)).astype(np.float32)
    return reg, sub


class _Chunked(np.ndarray):
    """A numpy array that reports dask-style `.chunks`, as npy_to_dask would."""

    @classmethod
    def make(cls, a, chunks):
        v = a.view(cls)
        v.chunks = chunks
        return v


def test_detection_stride_is_read_off_the_data_not_the_params():
    # TC040: params['fs'] is the PLANE rate, so 2*round(fs/tau) would say 46.
    # The movie says 3: 800 volumes per batch -> 266 bins (the tail is dropped).
    reg = np.zeros((7, 1788, 2, 2), np.float32)
    sub = _Chunked.make(np.zeros((594, 7, 2, 2), np.float32), ((266, 266, 62),))
    job = _FakeJob("/nonexistent", params={"t_batch_size": 800, "detection_timebin": 46})
    assert MV.detection_stride(job, {MV.REGISTERED: reg, MV.DETECTION: sub}) == 3


def test_detection_stride_falls_back_to_the_trace_length():
    sub = np.zeros((10, 1, 2, 2), np.float32)       # no registered movie, no chunks
    job = _FakeJob("/nonexistent", params={})
    assert MV.detection_stride(job, {MV.DETECTION: sub}, nt_trace=100) == 10


def test_write_snippets_lays_out_files_and_time_mapping(tmp_path):
    reg, sub = _movies(nt=100, timebin=5)           # 100 volumes -> 20 bins
    job = _FakeJob(tmp_path, reg, sub)
    vdir = str(tmp_path / "viewer")
    spec = MV.normalize_spec({"n_frames": 3, "start": 10})
    meta = MV.write_movie_snippets(job, vdir, spec, nz=2, ny=8, nx=8)

    assert set(meta) == {MV.REGISTERED, MV.DETECTION}
    for key in meta:
        assert meta[key]["n_frames"] == 3
        for z in range(2):
            for t in range(3):
                assert os.path.exists(os.path.join(vdir, "movies", key, f"z{z:02d}_t{t:04d}.jpg"))

    # frame -> sample on the trace time axis
    assert meta[MV.REGISTERED]["t0"] == 10 and meta[MV.REGISTERED]["stride"] == 1
    # mov_sub is indexed in bins: start=10 volumes -> bin 2 -> sample 10, stride 5
    assert meta[MV.DETECTION]["t0"] == 10 and meta[MV.DETECTION]["stride"] == 5


def test_window_slides_back_when_the_movie_is_short(tmp_path):
    reg, _ = _movies(nt=6)
    job = _FakeJob(tmp_path, reg, None)
    meta = MV.write_movie_snippets(job, str(tmp_path / "v"),
                                   MV.normalize_spec({"n_frames": 4, "start": 5}), 2, 8, 8)
    assert meta[MV.REGISTERED]["n_frames"] == 4
    assert meta[MV.REGISTERED]["t0"] == 2           # 5 would leave only 1 frame


def test_missing_mov_sub_is_not_an_error(tmp_path):
    reg, _ = _movies()
    job = _FakeJob(tmp_path, reg, None)             # no mov_sub/ dir
    meta = MV.write_movie_snippets(job, str(tmp_path / "v"), MV.normalize_spec(2), 2, 8, 8)
    assert set(meta) == {MV.REGISTERED}
    assert any("no mov_sub" in m for m in job.lines)


def test_no_movies_at_all_returns_empty(tmp_path):
    job = _FakeJob(tmp_path, None, None)
    assert MV.write_movie_snippets(job, str(tmp_path / "v"), MV.normalize_spec(2), 2, 8, 8) == {}


def test_misaligned_fov_is_skipped_not_stretched(tmp_path):
    """A movie whose FOV differs from the ROI grid would put ROIs on wrong pixels."""
    reg, sub = _movies(ny=8, nx=8)
    job = _FakeJob(tmp_path, reg, sub)
    meta = MV.write_movie_snippets(job, str(tmp_path / "v"), MV.normalize_spec(2),
                                   nz=2, ny=16, nx=16)
    assert meta == {}
    assert any("would not align" in m for m in job.lines)


def test_frames_are_scaled_once_per_plane_so_playback_does_not_flicker(tmp_path):
    """A frame twice as bright must *look* twice as bright, not be renormalised."""
    from PIL import Image
    nz, nt, ny, nx = 1, 2, 8, 8
    reg = np.zeros((nz, nt, ny, nx), np.float32)
    reg[0, 0] = np.linspace(0, 1, ny * nx).reshape(ny, nx)
    reg[0, 1] = reg[0, 0] * 0.5
    job = _FakeJob(tmp_path, reg, None)
    vdir = str(tmp_path / "v")
    MV.write_movie_snippets(job, vdir, MV.normalize_spec({"n_frames": 2, "fmt": "png"}), 1, ny, nx)
    a = np.asarray(Image.open(os.path.join(vdir, "movies", "registered", "z00_t0000.png")), float)
    b = np.asarray(Image.open(os.path.join(vdir, "movies", "registered", "z00_t0001.png")), float)
    assert b.max() < 0.6 * a.max()
