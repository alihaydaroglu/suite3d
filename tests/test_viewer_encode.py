"""Round-trip tests for the HTML viewer's encoders."""
import base64
import numpy as np
import pytest

from suite3d.viewer.encode import b64, npy_bytes, quantize_traces, rle
from suite3d.viewer.curation import apply_curation, stat_table


def _dec(s, dt):
    return np.frombuffer(base64.b64decode(s), dtype=dt)


def _fake_stats(rng, n=40, nz=5, ny=60, nx=70):
    stats = []
    for _ in range(n):
        z0 = rng.integers(0, nz)
        y0, x0 = rng.integers(0, ny - 8), rng.integers(0, nx - 8)
        zz, yy, xx = [], [], []
        for dz in range(rng.integers(1, min(3, nz - z0) + 1)):
            for dy in range(rng.integers(2, 7)):
                w = int(rng.integers(1, 8))
                for dx in range(w):
                    zz.append(z0 + dz); yy.append(y0 + dy); xx.append(x0 + dx)
        lam = rng.random(len(zz)).astype(np.float32) + 1e-3
        stats.append({"coords": np.array([zz, yy, xx]), "lam": lam,
                      "peak_val": float(rng.random()),
                      "vox_snrs": rng.random(len(zz))})
    return stats


def test_rle_roundtrip_reproduces_every_voxel():
    rng = np.random.default_rng(0)
    stats = _fake_stats(rng)
    m = rle(stats)
    for i, s in enumerate(stats):
        got = set()
        for r in range(m["off"][i], m["off"][i + 1]):
            for k in range(m["len"][r]):
                got.add((int(m["z"][r]), int(m["y"][r]), int(m["x0"][r] + k)))
        want = set(map(tuple, np.asarray(s["coords"]).T.tolist()))
        assert got == want, f"roi {i}: {len(got)} vs {len(want)} voxels"


def test_rle_splits_runs_longer_than_255():
    coords = np.array([[0] * 600, [3] * 600, list(range(600))])
    stats = [{"coords": coords, "lam": np.ones(600, np.float32), "peak_val": 1.0}]
    m = rle(stats)
    assert m["n_runs"] == 3                       # 255 + 255 + 90
    assert m["len"].tolist() == [255, 255, 90]
    assert m["x0"].tolist() == [0, 255, 510]


def test_quantize_int16_is_visually_lossless():
    rng = np.random.default_rng(1)
    F = rng.normal(500, 80, size=(17, 900)).astype(np.float32)
    q, sc, off = quantize_traces(F, "int16")
    rec = q.astype(np.float64) * sc[:, None] + off[:, None]
    span = F.max(axis=1) - F.min(axis=1)
    err = np.abs(rec - F).max(axis=1) / span
    assert (err < 1e-4).all(), err.max()


def test_quantize_handles_flat_traces():
    F = np.full((3, 50), 7.0, np.float32)
    q, sc, off = quantize_traces(F, "int16")
    rec = q.astype(np.float64) * sc[:, None] + off[:, None]
    assert np.allclose(rec, 7.0)


def test_npy_bytes_is_loadable():
    a = np.array([True, False, True])
    import io
    assert np.array_equal(np.load(io.BytesIO(npy_bytes(a))), a)


def test_curation_manual_overrides_filters():
    tbl = {"npix": np.array([10.0, 100.0, 1000.0])}
    keep = apply_curation(tbl, {"npix": [50, 500]}, {})
    assert keep.tolist() == [False, True, False]
    keep = apply_curation(tbl, {"npix": [50, 500]}, {0: 1, 1: 0})
    assert keep.tolist() == [True, False, False]


def test_curation_nan_never_fails_a_filter():
    tbl = {"peak_val": np.array([np.nan, 0.5])}
    keep = apply_curation(tbl, {"peak_val": [1.0, 2.0]}, {})
    assert keep.tolist() == [True, False]


def test_stat_table_shapes():
    stats = _fake_stats(np.random.default_rng(2), n=9)
    t = stat_table(stats)
    assert set(t) == {"npix", "zspan", "peak_val", "vox_snr"}
    assert all(len(v) == 9 for v in t.values())
