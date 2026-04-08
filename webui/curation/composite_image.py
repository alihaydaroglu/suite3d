from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from typing import List, Dict, Tuple, Optional


class NeighborhoodFinder:
    """Finds neighbouring cells via KD-tree with anisotropic z-scaling."""

    def __init__(
        self,
        stats: List[Dict],
        radius_yx: float = 15.0,
        radius_z: float = 3.0,
    ):
        self.stats = stats
        self.radius_yx = radius_yx

        # Scale z so Euclidean ball → anisotropic ellipsoid
        self.z_scale = radius_yx / max(radius_z, 1e-6)
        centers = np.array([s["med"] for s in stats], dtype=np.float64)  # (N,3) z,y,x
        self.centers = centers

        scaled = centers.copy()
        scaled[:, 0] *= self.z_scale
        self.tree = cKDTree(scaled)

    def find_neighbors(self, cell_idx: int) -> List[int]:
        q = self.centers[cell_idx].copy()
        q[0] *= self.z_scale
        indices = self.tree.query_ball_point(q, r=self.radius_yx)
        return [i for i in indices if i != cell_idx]


class CompositeImageBuilder:
    """
    Builds 224×224 RGB composite per cell.

    Layout (2×2, each quadrant 112×112):
        ┌────────────┬────────────┐
        │ XY max-proj │ XZ max-proj│
        ├────────────┼────────────┤
        │ XY mid-z    │ XZ mid-y   │
        └────────────┴────────────┘

    R = this cell's footprint, G = neighbour footprints, B = 0.
    """

    def __init__(
        self,
        stats: List[Dict],
        box_size: Tuple[int, int, int] = (5, 20, 20),
        output_size: int = 224,
    ):
        self.stats = stats
        self.nbz, self.nby, self.nbx = box_size
        self.output_size = output_size
        self.q = output_size // 2  # quadrant size = 112

        self.nf = NeighborhoodFinder(
            stats,
            radius_yx=float(max(self.nby, self.nbx)),
            radius_z=float(self.nbz),
        )

    # ------------------------------------------------------------------
    def _footprint_in_frame(
        self, cell_idx: int, ref_med: Tuple[int, int, int]
    ) -> np.ndarray:
        """Reconstruct one cell's footprint centred on *ref_med*."""
        fp = np.zeros((self.nbz, self.nby, self.nbx), dtype=np.float32)
        s = self.stats[cell_idx]
        if "coords" not in s or "lam" not in s:
            return fp
        coords, lam = s["coords"], s["lam"]
        if len(coords) != 3 or len(lam) == 0:
            return fp

        cz, cy, cx = self.nbz // 2, self.nby // 2, self.nbx // 2
        rz = coords[0] - ref_med[0] + cz
        ry = coords[1] - ref_med[1] + cy
        rx = coords[2] - ref_med[2] + cx
        ok = (
            (rz >= 0) & (rz < self.nbz)
            & (ry >= 0) & (ry < self.nby)
            & (rx >= 0) & (rx < self.nbx)
        )
        if ok.any():
            fp[rz[ok].astype(int), ry[ok].astype(int), rx[ok].astype(int)] = lam[ok]
        return fp

    # ------------------------------------------------------------------
    @staticmethod
    def _upscale(img: np.ndarray, target: int) -> np.ndarray:
        """Nearest-neighbour upscale a small 2-D array, centred in *target×target*."""
        h, w = img.shape
        ry = max(1, target // h)
        rx = max(1, target // w)
        up = np.repeat(np.repeat(img, ry, axis=0), rx, axis=1)
        out = np.zeros((target, target), dtype=np.float32)
        ch, cw = min(up.shape[0], target), min(up.shape[1], target)
        oy, ox = (target - ch) // 2, (target - cw) // 2
        out[oy : oy + ch, ox : ox + cw] = up[:ch, :cw]
        return out

    @staticmethod
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255.0
        return arr.astype(np.uint8)

    # ------------------------------------------------------------------
    def build_composite(self, cell_idx: int) -> np.ndarray:
        """Return (224, 224, 3) uint8 composite for one cell."""
        ref = tuple(self.stats[cell_idx]["med"])

        self_fp = self._footprint_in_frame(cell_idx, ref)
        nb_fp = np.zeros_like(self_fp)
        for ni in self.nf.find_neighbors(cell_idx):
            nb_fp += self._footprint_in_frame(ni, ref)

        cz, cy = self.nbz // 2, self.nby // 2
        q = self.q
        up = self._upscale

        # 4 views × 2 channels
        views_self = [
            np.max(self_fp, axis=0),        # XY max-proj  (nby, nbx)
            np.max(self_fp, axis=1),         # XZ max-proj  (nbz, nbx)
            self_fp[cz, :, :],               # XY mid-z
            self_fp[:, cy, :],               # XZ mid-y
        ]
        views_nb = [
            np.max(nb_fp, axis=0),
            np.max(nb_fp, axis=1),
            nb_fp[cz, :, :],
            nb_fp[:, cy, :],
        ]

        canvas = np.zeros((self.output_size, self.output_size, 3), dtype=np.float32)
        # positions: (row_start, col_start)
        positions = [(0, 0), (0, q), (q, 0), (q, q)]
        for (r, c), vs, vn in zip(positions, views_self, views_nb):
            canvas[r : r + q, c : c + q, 0] = up(vs, q)
            canvas[r : r + q, c : c + q, 1] = up(vn, q)

        return self._to_uint8(canvas)

    # ------------------------------------------------------------------
    def build_all(
        self,
        cell_indices: Optional[np.ndarray] = None,
        progress_every: int = 1000,
    ) -> np.ndarray:
        """Return (N, 224, 224, 3) uint8 array."""
        if cell_indices is None:
            cell_indices = np.arange(len(self.stats))
        N = len(cell_indices)
        out = np.zeros((N, self.output_size, self.output_size, 3), dtype=np.uint8)
        for i, cidx in enumerate(cell_indices):
            out[i] = self.build_composite(cidx)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  Composed {i + 1}/{N} images")
        print(f"  Composed {N}/{N} images")
        return out
