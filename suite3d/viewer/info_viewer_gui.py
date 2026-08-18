from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QShortcut,
    QToolTip,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QKeySequence

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator


DISPLAY_KEYS = ("max_img", "mean_img", "vmap", "vmap_raw")
RECORDING_DISPLAY_KEY = "registered movie"
BLACK_DISPLAY_KEY = "Black"
TRACE_FILES = {
    "F": "F.npy",
    "Fneu": "Fneu.npy",
    "spks": "spks.npy",
}
SKEW_STAT_KEYS = ("trace_skew", "skewness", "skew")
TRACE_SKEW_CHUNK_ROIS = 1024
SURFACE_SMOOTHING_SIGMA = (0.35, 0.65, 0.65)
GUI_BG = "#4a4a4a"
PANEL_BG = "#5a5a5a"
PLOT_BG = "#6a6a6a"
TEXT_FG = "#f2f2f2"
GRID_FG = "#d8d8d8"


class Suite3DAnalysisWorker(QThread):
    progress_changed = pyqtSignal(int, str)
    log_message = pyqtSignal(str)
    finished_with_status = pyqtSignal(bool, str, str)

    def __init__(
        self,
        tif_dir: Path,
        output_dir: Path,
        job_id: str,
        params: dict,
        overwrite: bool,
        test_batch_only: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.tif_dir = tif_dir
        self.output_dir = output_dir
        self.job_id = job_id
        self.params = params
        self.overwrite = overwrite
        self.test_batch_only = test_batch_only

    def run(self) -> None:
        try:
            from suite3d import io
            from suite3d.job import Job

            self.progress_changed.emit(2, "Finding TIFF files")
            tifs = io.get_tif_paths(self.tif_dir)
            if not tifs:
                raise RuntimeError(f"No TIFF files found in {self.tif_dir}")
            tifs = [str(Path(tif)) for tif in tifs]
            if self.test_batch_only:
                tifs = tifs[:1]
                self.log_message.emit("Test batch: using only the first TIFF file.")

            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.progress_changed.emit(5, "Creating Suite3D job")
            job = Job(
                self.output_dir,
                self.job_id,
                params=self.params,
                tifs=tifs,
                overwrite=self.overwrite,
                verbosity=10,
            )
            original_log = job.log

            def gui_log(message="", level=1, logfile=True, log_mem_usage=False, tic=False, toc=False, **kwargs):
                self.log_message.emit(str(message))
                return original_log(
                    message,
                    level=level,
                    logfile=logfile,
                    log_mem_usage=log_mem_usage,
                    tic=tic,
                    toc=toc,
                    **kwargs,
                )

            job.log = gui_log

            stages = [
                ("Initialization", 10, 22, job.run_init_pass),
                ("Registration", 22, 45, job.register),
                ("Correlation map", 45, 65, self.run_corrmap_stage(job)),
                ("ROI segmentation", 65, 78, job.segment_rois),
                ("Neuropil masks", 78, 87, job.compute_npil_masks),
                ("Trace extraction/deconvolution", 87, 98, self.run_extraction_stage(job)),
            ]
            for stage_name, start_pct, end_pct, stage_func in stages:
                self.progress_changed.emit(start_pct, stage_name)
                self.log_message.emit(f"Starting {stage_name}")
                stage_func()
                self.progress_changed.emit(end_pct, f"Finished {stage_name}")

            rois_info = Path(job.dirs.get("rois", "")) / "info.npy"
            self.progress_changed.emit(100, "Analysis complete")
            self.finished_with_status.emit(True, "Analysis complete", str(rois_info))
        except Exception as exc:
            self.log_message.emit(traceback.format_exc())
            self.finished_with_status.emit(False, f"{type(exc).__name__}: {exc}", "")

    def run_corrmap_stage(self, job):
        def _run():
            iter_limit = 1 if self.test_batch_only else None
            job.calculate_corr_map(iter_limit=iter_limit)

        return _run

    def run_extraction_stage(self, job):
        def _run():
            n_frames = None
            if self.test_batch_only:
                n_frames = int(job.params.get("t_batch_size", 500))
            job.extract_and_deconvolve(n_frames=n_frames)

        return _run


class Roi3DWindow(QDialog):
    def __init__(
        self,
        roi_idx: int,
        stat: dict,
        rois_dir: Path | None = None,
        parent=None,
        mask_kind: str = "roi",
    ) -> None:
        super().__init__(parent)
        self.mask_kind = "neuropil" if mask_kind == "neuropil" else "roi"
        self.setWindowTitle(f"ROI {roi_idx} {self.mask_label()} 3D view")
        self.roi_idx = roi_idx
        self.stat = stat
        self.rois_dir = rois_dir
        self.correlation_cache: np.ndarray | None = None
        self.voxel_size_um = self.load_voxel_size_um()

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Render"))
        self.render_mode_combo = QComboBox()
        if self.mask_kind == "neuropil":
            self.render_mode_combo.addItems(["All neuropil voxels", "Smoothed surface"])
        else:
            self.render_mode_combo.addItems(["All ROI voxels", "Smoothed surface"])
        self.render_mode_combo.currentTextChanged.connect(self.plot_roi)
        controls.addWidget(self.render_mode_combo)
        controls.addSpacing(16)
        controls.addWidget(QLabel("Color"))
        self.color_mode_combo = QComboBox()
        if self.mask_kind == "neuropil":
            self.color_mode_combo.addItems(["Neuropil mask"])
        else:
            self.color_mode_combo.addItems(["ROI spatial weights", "Pixel correlation"])
        self.color_mode_combo.currentTextChanged.connect(self.plot_roi)
        controls.addWidget(self.color_mode_combo)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.figure = Figure(figsize=(10, 8.5), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet(f"background-color: {GUI_BG};")
        self.canvas.setMinimumSize(900, 760)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, stretch=1)
        self.resize(1100, 950)

        self.plot_roi()

    def plot_roi(self, *_args) -> None:
        coords = self.current_mask_coords()
        if coords is None:
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                f"ROI {self.roi_idx} has no saved {self.mask_label().lower()} coordinates",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        z, y, x = [np.asarray(c, dtype=float) for c in coords]
        x_display, y_display, z_display = self.display_xyz(x, y, z)
        weights = self.current_mask_weights(z.size)
        values = weights
        colorbar_label = "Neuropil mask" if self.mask_kind == "neuropil" else "ROI's spatial weights"

        if self.mask_kind == "roi" and self.color_mode_combo.currentText() == "Pixel correlation":
            correlations = self.pixel_correlation_values(z, y, x, weights)
            if correlations is not None:
                values = correlations
                colorbar_label = "Pixel correlation with ROI trace"
            else:
                colorbar_label = "ROI's spatial weights"

        self.figure.clear()
        self.figure.patch.set_facecolor(GUI_BG)
        ax = self.figure.add_subplot(111, projection="3d")
        ax.set_facecolor(PLOT_BG)
        render_mode = self.render_mode_combo.currentText()
        if render_mode == "Smoothed surface":
            mappable = self.plot_smoothed_roi_surface(ax, x, y, z, values, geometry_weights=weights)
        else:
            mappable = self.plot_roi_points(ax, x_display, y_display, z_display, values, geometry_weights=weights)
        self.style_3d_axes(ax)
        ax.set_title(f"{self.roi_title(self.roi_idx, self.stat, weights.size)}; {render_mode}", color=TEXT_FG)
        self.set_3d_limits(ax, x_display, y_display, z_display)
        if mappable is not None:
            cbar = self.figure.colorbar(mappable, ax=ax, label=colorbar_label, shrink=0.75)
            cbar.ax.yaxis.label.set_color(TEXT_FG)
            cbar.ax.tick_params(colors=TEXT_FG)
        self.canvas.draw_idle()

    def current_mask_coords(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        coord_key = "npcoords" if self.mask_kind == "neuropil" else "coords"
        coords = self.stat.get(coord_key)
        if coords is None or len(coords) != 3:
            return None
        z, y, x = [np.asarray(c) for c in coords]
        if z.size == 0 or y.size != z.size or x.size != z.size:
            return None
        return z, y, x

    def current_mask_weights(self, n_voxels: int) -> np.ndarray:
        if self.mask_kind == "roi":
            lam = self.stat.get("lam")
            if lam is not None:
                weights = np.asarray(lam, dtype=float)
                if weights.size == n_voxels:
                    return weights
        return np.ones(n_voxels, dtype=float)

    def mask_label(self) -> str:
        return "Neuropil" if self.mask_kind == "neuropil" else "ROI"

    def load_voxel_size_um(self) -> tuple[float, float, float] | None:
        if self.rois_dir is None:
            return None

        params_path = self.rois_dir.parent / "params.npy"
        if not params_path.exists():
            return None

        try:
            params = np.load(params_path, allow_pickle=True).item()
        except Exception:
            return None
        if not isinstance(params, dict):
            return None

        voxel_size = params.get("voxel_size_um")
        if voxel_size is None:
            return None

        try:
            voxel_size = np.asarray(voxel_size, dtype=float).ravel()
        except Exception:
            return None
        if voxel_size.size < 3:
            return None

        z_um_per_plane = float(voxel_size[-3])
        y_um_per_pixel = float(voxel_size[-2])
        x_um_per_pixel = float(voxel_size[-1])
        if not np.isfinite(z_um_per_plane) or not np.isfinite(y_um_per_pixel) or not np.isfinite(x_um_per_pixel):
            return None
        if z_um_per_plane <= 0 or y_um_per_pixel <= 0 or x_um_per_pixel <= 0:
            return None
        return z_um_per_plane, y_um_per_pixel, x_um_per_pixel

    def display_xyz(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.voxel_size_um is None:
            return x, y, z
        z_um_per_plane, y_um_per_pixel, x_um_per_pixel = self.voxel_size_um
        return x * x_um_per_pixel, y * y_um_per_pixel, z * z_um_per_plane

    def roi_title(self, roi_idx: int, stat: dict, n_voxels: int) -> str:
        return f"ROI {roi_idx} {self.mask_label().lower()}; {n_voxels} voxels"

    def plot_smoothed_roi_surface(
        self,
        ax,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        values: np.ndarray,
        geometry_weights: np.ndarray | None = None,
    ):
        geometry_values = values if geometry_weights is None else geometry_weights
        volume, origin = self.roi_weight_volume(x, y, z, geometry_values)
        color_volume, _color_origin = self.roi_weight_volume(x, y, z, values)
        if volume is None or color_volume is None:
            x_display, y_display, z_display = self.display_xyz(x, y, z)
            return self.plot_roi_points(ax, x_display, y_display, z_display, values, geometry_weights=geometry_weights)

        try:
            from matplotlib import cm
            from matplotlib.colors import Normalize
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from scipy.ndimage import gaussian_filter
            from skimage.measure import marching_cubes

            smoothed = gaussian_filter(volume, sigma=SURFACE_SMOOTHING_SIGMA)
            smoothed_colors = gaussian_filter(color_volume, sigma=SURFACE_SMOOTHING_SIGMA)
            positive = smoothed[smoothed > 0]
            if positive.size == 0:
                x_display, y_display, z_display = self.display_xyz(x, y, z)
                return self.plot_roi_points(ax, x_display, y_display, z_display, values, geometry_weights=geometry_weights)

            level = max(float(smoothed.max()) * 0.18, float(np.percentile(positive, 45)))
            verts, faces, _normals, vertex_geometry_values = marching_cubes(smoothed, level=level)
            vertex_color_values = self.sample_volume_nearest(smoothed_colors, verts)
            verts[:, 0] += origin[0]
            verts[:, 1] += origin[1]
            verts[:, 2] += origin[2]

            polygons = verts[faces][:, :, [2, 1, 0]]
            if self.voxel_size_um is not None:
                z_um_per_plane, y_um_per_pixel, x_um_per_pixel = self.voxel_size_um
                polygons[:, :, 0] *= x_um_per_pixel
                polygons[:, :, 1] *= y_um_per_pixel
                polygons[:, :, 2] *= z_um_per_plane
            finite_color_values = vertex_color_values[np.isfinite(vertex_color_values)]
            if finite_color_values.size == 0 or float(finite_color_values.max()) <= float(finite_color_values.min()):
                finite_color_values = vertex_geometry_values[np.isfinite(vertex_geometry_values)]
            if finite_color_values.size == 0 or float(finite_color_values.max()) <= float(finite_color_values.min()):
                finite_color_values = positive

            norm = Normalize(vmin=float(finite_color_values.min()), vmax=float(finite_color_values.max()))
            face_values = vertex_color_values[faces].mean(axis=1)
            face_values = np.nan_to_num(face_values, nan=float(finite_color_values.min()))
            colors = cm.plasma(norm(face_values))
            colors[:, 3] = self.alpha_from_values(norm(face_values), min_alpha=0.28)

            surface = Poly3DCollection(polygons, facecolors=colors, linewidths=0.05, edgecolors=(1, 1, 1, 0.12))
            ax.add_collection3d(surface)
            center_weights = np.maximum(np.asarray(geometry_values, dtype=float), 0)
            if not np.any(center_weights > 0):
                center_weights = np.ones_like(center_weights)
            center_x = float(np.average(x, weights=center_weights))
            center_y = float(np.average(y, weights=center_weights))
            center_z = float(np.average(z, weights=center_weights))
            if self.voxel_size_um is not None:
                z_um_per_plane, y_um_per_pixel, x_um_per_pixel = self.voxel_size_um
                center_x *= x_um_per_pixel
                center_y *= y_um_per_pixel
                center_z *= z_um_per_plane
            ax.scatter(
                [center_x],
                [center_y],
                [center_z],
                color="white",
                s=18,
                alpha=0.85,
            )

            mappable = cm.ScalarMappable(norm=norm, cmap="plasma")
            mappable.set_array([])
            return mappable
        except Exception:
            return self.plot_roi_voxels(ax, volume, origin)

    def sample_volume_nearest(self, volume: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        indices = np.rint(vertices).astype(int)
        indices[:, 0] = np.clip(indices[:, 0], 0, volume.shape[0] - 1)
        indices[:, 1] = np.clip(indices[:, 1], 0, volume.shape[1] - 1)
        indices[:, 2] = np.clip(indices[:, 2], 0, volume.shape[2] - 1)
        return volume[indices[:, 0], indices[:, 1], indices[:, 2]]

    def roi_weight_volume(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        values: np.ndarray,
    ) -> tuple[np.ndarray | None, tuple[int, int, int]]:
        if x.size == 0 or y.size == 0 or z.size == 0 or values.size == 0:
            return None, (0, 0, 0)

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(values)
        if not np.any(valid):
            return None, (0, 0, 0)

        xi = np.rint(x[valid]).astype(int)
        yi = np.rint(y[valid]).astype(int)
        zi = np.rint(z[valid]).astype(int)
        values = np.asarray(values[valid], dtype=float)
        if float(values.max()) <= 0:
            values = np.ones_like(values, dtype=float)

        pad = 2
        z0 = int(zi.min()) - pad
        y0 = int(yi.min()) - pad
        x0 = int(xi.min()) - pad
        shape = (
            int(zi.max() - zi.min()) + 1 + 2 * pad,
            int(yi.max() - yi.min()) + 1 + 2 * pad,
            int(xi.max() - xi.min()) + 1 + 2 * pad,
        )
        volume = np.zeros(shape, dtype=np.float32)
        np.maximum.at(volume, (zi - z0, yi - y0, xi - x0), values.astype(np.float32))
        return volume, (z0, y0, x0)

    def plot_roi_voxels(self, ax, volume: np.ndarray, origin: tuple[int, int, int]):
        try:
            from matplotlib import cm
            from matplotlib.colors import Normalize

            level = max(float(volume.max()) * 0.18, 0.01)
            filled = volume >= level
            if not np.any(filled):
                return None

            z_idx, y_idx, x_idx = np.indices(np.array(filled.shape) + 1)
            z_idx = z_idx + origin[0] - 0.5
            y_idx = y_idx + origin[1] - 0.5
            x_idx = x_idx + origin[2] - 0.5
            if self.voxel_size_um is not None:
                z_um_per_plane, y_um_per_pixel, x_um_per_pixel = self.voxel_size_um
                x_idx = x_idx * x_um_per_pixel
                y_idx = y_idx * y_um_per_pixel
                z_idx = z_idx * z_um_per_plane
            norm = Normalize(vmin=level, vmax=float(volume.max()))
            facecolors = cm.plasma(norm(volume))
            facecolors[..., 3] = self.alpha_from_values(norm(volume), min_alpha=0.28)
            ax.voxels(x_idx, y_idx, z_idx, filled, facecolors=facecolors, edgecolor=(1, 1, 1, 0.08))
            mappable = cm.ScalarMappable(norm=norm, cmap="plasma")
            mappable.set_array([])
            return mappable
        except Exception:
            return None

    def plot_roi_points(
        self,
        ax,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        values: np.ndarray,
        geometry_weights: np.ndarray | None = None,
    ):
        from matplotlib import cm
        from matplotlib.colors import Normalize

        values = np.asarray(values, dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            plot_values = np.zeros_like(values, dtype=float)
            norm = Normalize(vmin=0.0, vmax=1.0)
        else:
            fill_value = float(finite_values.min())
            plot_values = np.nan_to_num(values, nan=fill_value, posinf=float(finite_values.max()), neginf=fill_value)
            vmin = float(np.min(plot_values))
            vmax = float(np.max(plot_values))
            if vmax <= vmin:
                vmax = vmin + 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)

        point_weights = values if geometry_weights is None else np.asarray(geometry_weights, dtype=float)
        finite_weights = point_weights[np.isfinite(point_weights)]
        if finite_weights.size == 0:
            sizes = np.full(values.shape, 24.0)
        else:
            fill_weight = float(finite_weights.min())
            point_weights = np.nan_to_num(
                point_weights,
                nan=fill_weight,
                posinf=float(finite_weights.max()),
                neginf=fill_weight,
            )
            wmin = float(np.min(point_weights))
            wmax = float(np.max(point_weights))
            if wmax <= wmin:
                sizes = np.full(point_weights.shape, 28.0)
            else:
                sizes = 14.0 + 44.0 * ((point_weights - wmin) / (wmax - wmin))

        colors = cm.plasma(norm(plot_values))
        colors[:, 3] = self.alpha_from_values(norm(plot_values), min_alpha=0.32)
        ax.scatter(
            x,
            y,
            z,
            c=colors,
            s=sizes,
            edgecolors=(1, 1, 1, 0.18),
            linewidths=0.15,
            depthshade=False,
        )
        mappable = cm.ScalarMappable(norm=norm, cmap="plasma")
        mappable.set_array([])
        return mappable

    def alpha_from_values(self, normalized_values: np.ndarray, min_alpha: float = 0.3) -> np.ndarray:
        alpha = np.asarray(normalized_values, dtype=float)
        alpha = np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        return min_alpha + (1.0 - min_alpha) * alpha

    def style_3d_axes(self, ax) -> None:
        if self.voxel_size_um is None:
            ax.set_xlabel("x (pixels)")
            ax.set_ylabel("y (pixels)")
            ax.set_zlabel("z plane")
        else:
            ax.set_xlabel("x (microns)")
            ax.set_ylabel("y (microns)")
            ax.set_zlabel("z (microns)")
        ax.xaxis.label.set_color(TEXT_FG)
        ax.yaxis.label.set_color(TEXT_FG)
        ax.zaxis.label.set_color(TEXT_FG)
        ax.tick_params(colors=TEXT_FG)
        try:
            ax.zaxis.set_major_locator(MaxNLocator(5))
        except Exception:
            pass

    def pixel_correlation_values(
        self,
        z: np.ndarray,
        y: np.ndarray,
        x: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray | None:
        if self.correlation_cache is not None:
            return self.correlation_cache
        if self.rois_dir is None:
            return None

        recording_dir = self.rois_dir.parent / "registered_fused_data"
        files = sorted(recording_dir.glob("fused_reg_data*.npy"))
        if not files:
            return None

        zi = np.rint(z).astype(int)
        yi = np.rint(y).astype(int)
        xi = np.rint(x).astype(int)
        valid = np.isfinite(z) & np.isfinite(y) & np.isfinite(x) & np.isfinite(weights)
        if not np.any(valid):
            return None

        zi = zi[valid]
        yi = yi[valid]
        xi = xi[valid]
        weights = np.asarray(weights[valid], dtype=float)
        weights = np.maximum(weights, 0)
        if not np.any(weights > 0):
            weights = np.ones_like(weights)

        trace_chunks = []
        for path in files:
            try:
                movie = np.load(path, mmap_mode="r", allow_pickle=False)
            except Exception:
                continue
            if movie.ndim != 4:
                continue
            nz, nt, ny, nx = movie.shape
            in_bounds = (
                (zi >= 0)
                & (zi < nz)
                & (yi >= 0)
                & (yi < ny)
                & (xi >= 0)
                & (xi < nx)
            )
            if not np.any(in_bounds):
                continue

            chunk = np.zeros((zi.size, nt), dtype=np.float32)
            for voxel_idx in np.flatnonzero(in_bounds):
                chunk[voxel_idx] = np.asarray(
                    movie[zi[voxel_idx], :, yi[voxel_idx], xi[voxel_idx]],
                    dtype=np.float32,
                )
            trace_chunks.append(chunk)

        if not trace_chunks:
            return None

        traces = np.concatenate(trace_chunks, axis=1)
        roi_trace = np.average(traces, axis=0, weights=weights)
        traces_centered = traces - traces.mean(axis=1, keepdims=True)
        roi_centered = roi_trace - roi_trace.mean()
        denom = np.sqrt((traces_centered**2).sum(axis=1) * float((roi_centered**2).sum()))
        corr_valid = np.divide(
            (traces_centered * roi_centered).sum(axis=1),
            denom,
            out=np.zeros(traces.shape[0], dtype=np.float32),
            where=denom > 0,
        )
        corr_valid = np.clip(corr_valid, 0.0, 1.0)

        correlations = np.zeros_like(np.asarray(z, dtype=float), dtype=float)
        correlations[valid] = corr_valid
        self.correlation_cache = correlations
        return self.correlation_cache

    def set_3d_limits(self, ax, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        x_margin = max(float(x.max() - x.min()) * 0.1, 1.0)
        y_margin = max(float(y.max() - y.min()) * 0.1, 1.0)
        z_margin = max(float(z.max() - z.min()) * 0.1, 1.0)

        ax.set_xlim(float(x.min()) - x_margin, float(x.max()) + x_margin)
        ax.set_ylim(float(y.min()) - y_margin, float(y.max()) + y_margin)
        z_low = max(0.0, float(z.min()) - z_margin)
        z_high = float(z.max()) + z_margin
        ax.set_zlim(z_high, z_low)
        try:
            x_span = max(float(x.max() - x.min()), 1.0)
            y_span = max(float(y.max() - y.min()), 1.0)
            z_span = max(float(z.max() - z.min()), min(x_span, y_span) * 0.65, 1.0)
            ax.set_box_aspect(
                (
                    x_span,
                    y_span,
                    z_span,
                )
            )
        except AttributeError:
            pass


class InfoViewer(QMainWindow):
    def __init__(self, info_path: Path | None = None, load_default: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("Suite3D info.npy viewer")

        self.info_path: Path | None = None
        self.info: dict | None = None
        self.current_array: np.ndarray | None = None
        self.current_image_shape: tuple[int, int, int] | None = None
        self.stats: np.ndarray | None = None
        self.iscell: np.ndarray | None = None
        self.roi_colors: np.ndarray | None = None
        self.correlation_colors: np.ndarray | None = None
        self.correlation_values: np.ndarray | None = None
        self.correlation_seed_roi_idx: int | None = None
        self.correlation_color_limits: tuple[float, float] | None = None
        self.roi_color_overlay_mode: str | None = None
        self.roi_colorbar_label: str | None = None
        self.roi_colorbar_cmap_name = "coolwarm"
        self.roi_colorbar_center_zero = False
        self.roi_metrics: dict[str, np.ndarray] = {}
        self.curation_controls: dict[str, dict[str, object]] = {}
        self.curation_metric_checkboxes: dict[str, QCheckBox] = {}
        self.curation_updating = False
        self.curation_axes: dict[object, str] = {}
        self.curation_threshold_lines: dict[tuple[str, str], object] = {}
        self.curation_title_artists: dict[object, str] = {}
        self.curation_tooltip_key: str | None = None
        self.curation_drag: dict[str, object] | None = None
        self.current_id_maps: dict[object, np.ndarray] = {}
        self.overlay_cache: dict[tuple, tuple[np.ndarray | None, np.ndarray]] = {}
        self.current_axes = None
        self.image_axes: list[object] = []
        self.image_axes_panels: dict[object, bool] = {}
        self.image_axes_roles: dict[object, str] = {}
        self.selected_roi_idx: int | None = None
        self.selected_panel_accepted = True
        self.selected_mask_kind = "roi"
        self.manual_curation_overrides: dict[int, bool] = {}
        self.neuropil_rejected_roi_indices: set[int] = set()
        self.traces: dict[str, np.ndarray] = {}
        self.trace_paths: dict[str, Path] = {}
        self.trace_roi_axes: dict[str, int] = {}
        self.trace_cursor_lines: list[object] = []
        self.motion_shifts: np.ndarray | None = None
        self.motion_shift_paths: list[Path] = []
        self.trace_zoom_axes: object | None = None
        self.trace_pan_start: dict[str, object] | None = None
        self.drag_start: dict[str, object] | None = None
        self.drag_pixel_threshold = 4
        self.selection_rect = None
        self.curation_undo_stack: list[dict[str, object]] = []
        self.roi_3d_windows: list[Roi3DWindow] = []
        self.recording_files: list[Path] = []
        self.recording_frame_counts: list[int] = []
        self.recording_frame_starts: np.ndarray | None = None
        self.recording_shape: tuple[int, int, int] | None = None
        self.recording_chunk_cache: tuple[int, np.ndarray] | None = None
        self.plane_timer: QTimer | None = None
        self.frame_timer: QTimer | None = None
        self.analysis_worker: Suite3DAnalysisWorker | None = None

        self._build_ui()

        if info_path is not None:
            self.load_info(info_path)

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background-color: {GUI_BG};
                color: {TEXT_FG};
            }}
            QGroupBox {{
                background-color: {PANEL_BG};
                color: {TEXT_FG};
                border: 1px solid #777777;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 3px;
            }}
            QLabel, QCheckBox {{
                color: {TEXT_FG};
            }}
            QPushButton, QComboBox, QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
                background-color: #6b6b6b;
                color: {TEXT_FG};
                border: 1px solid #909090;
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QPushButton:hover, QComboBox:hover, QLineEdit:hover, QPlainTextEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
                background-color: #777777;
            }}
            QScrollArea {{
                background-color: {PANEL_BG};
                border: 1px solid #777777;
            }}
            QScrollArea > QWidget > QWidget {{
                background-color: {PANEL_BG};
            }}
            QPushButton:pressed {{
                background-color: #555555;
            }}
            QSlider::groove:horizontal {{
                background: #777777;
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: #d0d0d0;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            """
        )

        root = QVBoxLayout(central)

        file_row = QHBoxLayout()
        self.open_button = QPushButton("Open run folder")
        self.open_button.clicked.connect(self.open_directory_dialog)
        self.open_file_button = QPushButton("Open info.npy")
        self.open_file_button.clicked.connect(self.open_file_dialog)
        self.analyze_data_button = QPushButton("Analyze data")
        self.analyze_data_button.clicked.connect(self.show_analysis_dialog)
        self.file_label = QLabel("No file loaded")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        file_row.addWidget(self.open_button)
        file_row.addWidget(self.open_file_button)
        file_row.addWidget(self.analyze_data_button)
        file_row.addWidget(self.file_label, stretch=1)
        root.addLayout(file_row)

        controls = QGroupBox("Display")
        controls_layout = QGridLayout(controls)

        self.data_combo = QComboBox()
        self.data_combo.currentTextChanged.connect(self.on_display_changed)
        controls_layout.addWidget(QLabel("Image"), 0, 0)
        controls_layout.addWidget(self.data_combo, 0, 1)

        self.project_checkbox = QCheckBox("Project across planes")
        self.project_checkbox.stateChanged.connect(self.on_display_changed)
        controls_layout.addWidget(self.project_checkbox, 0, 2)

        self.projection_combo = QComboBox()
        self.projection_combo.addItems(["Max", "Mean"])
        self.projection_combo.currentTextChanged.connect(self.on_display_changed)
        self.projection_label = QLabel("Projection")
        controls_layout.addWidget(self.projection_label, 0, 3)
        controls_layout.addWidget(self.projection_combo, 0, 4)

        self.roi_color_combo = QComboBox()
        self.roi_color_combo.addItems(
            ["Random", "Correlation", "Skewness", "Voxel count", "Peak value", "Voxel SNR"]
        )
        self.roi_color_combo.currentTextChanged.connect(self.on_roi_color_mode_changed)
        self.roi_color_combo.setEnabled(False)
        controls_layout.addWidget(QLabel("ROI colors"), 1, 0)
        controls_layout.addWidget(self.roi_color_combo, 1, 1)

        self.masks_checkbox = QCheckBox("Show cell ROIs")
        self.masks_checkbox.stateChanged.connect(self.on_display_changed)
        controls_layout.addWidget(self.masks_checkbox, 1, 2)

        self.show_nonaccepted_checkbox = QCheckBox("Show right panel")
        self.show_nonaccepted_checkbox.setChecked(True)
        self.show_nonaccepted_checkbox.stateChanged.connect(self.on_display_changed)
        controls_layout.addWidget(self.show_nonaccepted_checkbox, 1, 3)

        self.secondary_panel_combo = QComboBox()
        self.secondary_panel_combo.addItems(["Non-accepted cells", "Accepted neuropil"])
        self.secondary_panel_combo.currentTextChanged.connect(self.on_secondary_panel_mode_changed)
        self.secondary_panel_combo.setEnabled(False)
        controls_layout.addWidget(QLabel("Right panel"), 1, 4)
        controls_layout.addWidget(self.secondary_panel_combo, 1, 5)

        self.motion_checkbox = QCheckBox("Motion correction")
        self.motion_checkbox.stateChanged.connect(self.on_motion_correction_toggled)
        self.motion_checkbox.setEnabled(False)
        controls_layout.addWidget(self.motion_checkbox, 1, 6)

        self.plane_slider = QSlider(Qt.Horizontal)
        self.plane_slider.setMinimum(0)
        self.plane_slider.setMaximum(0)
        self.plane_slider.setTracking(False)
        self.plane_slider.sliderMoved.connect(self.on_plane_slider_moved)
        self.plane_slider.valueChanged.connect(self.on_plane_changed)
        self.plane_play_button = QPushButton("Play")
        self.plane_play_button.clicked.connect(self.toggle_plane_play)
        self.plane_play_button.setEnabled(False)
        self.plane_label = QLabel("Plane: -")
        controls_layout.addWidget(self.plane_label, 2, 0)
        controls_layout.addWidget(self.plane_play_button, 2, 1)
        controls_layout.addWidget(self.plane_slider, 2, 2, 1, 5)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setTracking(False)
        self.frame_slider.sliderMoved.connect(self.on_frame_slider_moved)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        self.frame_play_button = QPushButton("Play")
        self.frame_play_button.clicked.connect(self.toggle_frame_play)
        self.frame_play_button.setEnabled(False)
        self.frame_label = QLabel("Frame: -")
        controls_layout.addWidget(self.frame_label, 3, 0)
        controls_layout.addWidget(self.frame_play_button, 3, 1)
        controls_layout.addWidget(self.frame_slider, 3, 2, 1, 5)

        self.zoom_selected_button = QPushButton("Zoom selected ROI")
        self.zoom_selected_button.clicked.connect(self.zoom_to_selected_roi)
        controls_layout.addWidget(self.zoom_selected_button, 4, 0)

        self.reset_zoom_button = QPushButton("Reset zoom")
        self.reset_zoom_button.clicked.connect(self.reset_zoom)
        controls_layout.addWidget(self.reset_zoom_button, 4, 1)

        self.show_3d_button = QPushButton("Show selected ROI in 3D")
        self.show_3d_button.clicked.connect(self.show_selected_roi_3d)
        controls_layout.addWidget(self.show_3d_button, 4, 2)

        root.addWidget(controls)

        self.figure = Figure(figsize=(13.5, 8), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet(f"background-color: {GUI_BG};")
        self.canvas.setMinimumHeight(360)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.mpl_connect("button_press_event", self.on_canvas_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_canvas_motion)
        self.canvas.mpl_connect("button_release_event", self.on_canvas_release)
        self.canvas.mpl_connect("scroll_event", self.on_scroll_zoom)

        image_row = QHBoxLayout()
        image_row.addWidget(self.canvas, stretch=5)
        self.curation_group = self.build_curation_panel()
        image_row.addWidget(self.curation_group, stretch=2)
        root.addLayout(image_row, stretch=5)

        self.status_label = QLabel("")
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.status_label)

        self.trace_group = QGroupBox("F trace")
        trace_layout = QVBoxLayout(self.trace_group)
        trace_layout.setContentsMargins(4, 4, 4, 4)
        trace_layout.setSpacing(0)
        self.trace_figure = Figure(figsize=(12, 2.6), dpi=100, constrained_layout=False)
        self.trace_canvas = FigureCanvas(self.trace_figure)
        self.trace_canvas.setStyleSheet(f"background-color: {GUI_BG};")
        self.trace_canvas.setMinimumHeight(220)
        self.trace_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.trace_canvas.mpl_connect("scroll_event", self.on_trace_scroll_zoom)
        self.trace_canvas.mpl_connect("button_press_event", self.on_trace_press)
        self.trace_canvas.mpl_connect("motion_notify_event", self.on_trace_motion)
        self.trace_canvas.mpl_connect("button_release_event", self.on_trace_release)
        trace_layout.addWidget(self.trace_canvas)
        trace_controls = QHBoxLayout()
        trace_controls.setContentsMargins(0, 2, 0, 0)
        trace_controls.setSpacing(4)
        self.trace_full_view_checkbox = QCheckBox("Full view")
        self.trace_full_view_checkbox.setChecked(True)
        self.trace_full_view_checkbox.stateChanged.connect(self.on_trace_full_view_toggled)
        self.trace_visibility_checkboxes: dict[str, QCheckBox] = {}
        for label in ("F", "Neuropil", "Deconvolved"):
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.on_trace_visibility_changed)
            self.trace_visibility_checkboxes[label] = checkbox
        trace_controls.addStretch(1)
        trace_controls.addWidget(self.trace_full_view_checkbox)
        for checkbox in self.trace_visibility_checkboxes.values():
            trace_controls.addWidget(checkbox)
        trace_controls.addStretch(1)
        trace_layout.addLayout(trace_controls)
        root.addWidget(self.trace_group, stretch=2)
        self.clear_trace_plot("F trace: select an ROI")
        self.update_projection_controls_visibility()

        self.plane_timer = QTimer(self)
        self.plane_timer.setInterval(350)
        self.plane_timer.timeout.connect(self.advance_plane)
        self.frame_timer = QTimer(self)
        self.frame_timer.setInterval(200)
        self.frame_timer.timeout.connect(self.advance_frame)

        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo_curation_action)
        self.prev_roi_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.prev_roi_shortcut.activated.connect(lambda: self.navigate_selected_roi(-1))
        self.next_roi_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.next_roi_shortcut.activated.connect(lambda: self.navigate_selected_roi(1))

        self.analysis_dialog: QDialog | None = None

        self.resize(1500, 1000)

    def show_analysis_dialog(self) -> None:
        if self.analysis_dialog is None:
            self.analysis_dialog = self.build_analysis_dialog()
        self.analysis_dialog.show()
        self.analysis_dialog.raise_()
        self.analysis_dialog.activateWindow()

    def build_analysis_dialog(self) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle("Analyze data")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        paths_group = QGroupBox("Input and output")
        paths_layout = QGridLayout(paths_group)

        self.analysis_tif_dir_edit = QLineEdit()
        self.analysis_tif_dir_edit.setPlaceholderText("Directory containing raw 2P TIFF files")
        tif_browse = QPushButton("Browse")
        tif_browse.clicked.connect(self.browse_analysis_tif_dir)
        paths_layout.addWidget(QLabel("2P TIFF directory"), 0, 0)
        paths_layout.addWidget(self.analysis_tif_dir_edit, 0, 1)
        paths_layout.addWidget(tif_browse, 0, 2)

        self.analysis_output_dir_edit = QLineEdit()
        self.analysis_output_dir_edit.setPlaceholderText("Directory where the Suite3D job folder will be saved")
        output_browse = QPushButton("Browse")
        output_browse.clicked.connect(self.browse_analysis_output_dir)
        paths_layout.addWidget(QLabel("Saving directory"), 1, 0)
        paths_layout.addWidget(self.analysis_output_dir_edit, 1, 1)
        paths_layout.addWidget(output_browse, 1, 2)

        layout.addWidget(paths_group)

        options_group = QGroupBox("Run options")
        options_layout = QGridLayout(options_group)
        self.analysis_test_batch_checkbox = QCheckBox("Analyze only one test-batch")
        self.analysis_test_batch_checkbox.setToolTip(
            "Uses the first TIFF, runs one correlation-map batch, and limits extraction to t_batch_size frames."
        )
        options_layout.addWidget(self.analysis_test_batch_checkbox, 0, 0)
        layout.addWidget(options_group)

        params_group = QGroupBox("Suite3D parameter overrides")
        params_layout = QGridLayout(params_group)

        self.analysis_fs_spin = QDoubleSpinBox()
        self.analysis_fs_spin.setRange(0.001, 10000.0)
        self.analysis_fs_spin.setDecimals(3)
        self.analysis_fs_spin.setValue(4.0)
        params_layout.addWidget(QLabel("fs fallback"), 0, 0)
        params_layout.addWidget(self.analysis_fs_spin, 0, 1)
        self.analysis_fs_label = QLabel("fs: auto from TIFF metadata")
        params_layout.addWidget(self.analysis_fs_label, 0, 4, 1, 2)

        self.analysis_tau_spin = QDoubleSpinBox()
        self.analysis_tau_spin.setRange(0.001, 10000.0)
        self.analysis_tau_spin.setDecimals(3)
        self.analysis_tau_spin.setValue(1.3)
        params_layout.addWidget(QLabel("tau"), 0, 2)
        params_layout.addWidget(self.analysis_tau_spin, 0, 3)

        self.analysis_voxel_z_spin = QDoubleSpinBox()
        self.analysis_voxel_z_spin.setRange(0.001, 10000.0)
        self.analysis_voxel_z_spin.setDecimals(3)
        self.analysis_voxel_z_spin.setValue(15.0)
        params_layout.addWidget(QLabel("voxel_size_um z"), 1, 0)
        params_layout.addWidget(self.analysis_voxel_z_spin, 1, 1)
        self.analysis_xy_voxel_label = QLabel("y/x: auto from TIFF metadata")
        params_layout.addWidget(self.analysis_xy_voxel_label, 1, 2, 1, 4)

        self.analysis_planes_edit = QLineEdit("")
        self.analysis_planes_edit.setPlaceholderText("Blank = all planes; example: 0, 1, 2")
        params_layout.addWidget(QLabel("planes"), 2, 0)
        params_layout.addWidget(self.analysis_planes_edit, 2, 1, 1, 5)

        self.analysis_n_ch_tif_spin = QSpinBox()
        self.analysis_n_ch_tif_spin.setRange(1, 200)
        self.analysis_n_ch_tif_spin.setValue(1)
        params_layout.addWidget(QLabel("n_ch_tif"), 3, 0)
        params_layout.addWidget(self.analysis_n_ch_tif_spin, 3, 1)

        self.analysis_lbm_checkbox = QCheckBox("lbm")
        self.analysis_lbm_checkbox.setChecked(True)
        self.analysis_3d_reg_checkbox = QCheckBox("3d_reg")
        self.analysis_3d_reg_checkbox.setChecked(True)
        self.analysis_gpu_reg_checkbox = QCheckBox("gpu_reg")
        self.analysis_gpu_reg_checkbox.setChecked(True)
        params_layout.addWidget(self.analysis_lbm_checkbox, 3, 2)
        params_layout.addWidget(self.analysis_3d_reg_checkbox, 3, 3)
        params_layout.addWidget(self.analysis_gpu_reg_checkbox, 3, 4)

        self.analysis_neuropil_mask_method_combo = QComboBox()
        self.analysis_neuropil_mask_method_combo.addItems(["rectangular", "expanding"])
        params_layout.addWidget(QLabel("neuropil_mask_method"), 4, 0)
        params_layout.addWidget(self.analysis_neuropil_mask_method_combo, 4, 1, 1, 5)

        layout.addWidget(params_group)

        progress_group = QGroupBox("Analysis progress")
        progress_layout = QVBoxLayout(progress_group)
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setValue(0)
        self.analysis_stage_label = QLabel("Idle")
        self.analysis_log = QPlainTextEdit()
        self.analysis_log.setReadOnly(True)
        self.analysis_log.setMinimumHeight(180)
        progress_layout.addWidget(self.analysis_stage_label)
        progress_layout.addWidget(self.analysis_progress)
        progress_layout.addWidget(self.analysis_log, stretch=1)
        layout.addWidget(progress_group, stretch=2)

        run_row = QHBoxLayout()
        run_row.addStretch(1)
        self.analysis_run_button = QPushButton("Run analysis")
        self.analysis_run_button.clicked.connect(self.start_analysis)
        run_row.addWidget(self.analysis_run_button)
        layout.addLayout(run_row)
        dialog.resize(900, 720)
        return dialog

    def browse_analysis_tif_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select 2P TIFF directory")
        if directory:
            self.analysis_tif_dir_edit.setText(directory)

    def browse_analysis_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select saving directory")
        if directory:
            self.analysis_output_dir_edit.setText(directory)

    def first_analysis_tif(self, tif_dir: Path) -> Path:
        from suite3d import io

        tifs = io.get_tif_paths(tif_dir)
        if not tifs:
            raise ValueError(f"No TIFF files found in {tif_dir}.")
        return Path(tifs[0])

    def load_first_analysis_tif_metadata(self, tif_dir: Path) -> tuple[str, dict, Path]:
        import tifffile

        tif_path = self.first_analysis_tif(tif_dir)
        with tifffile.TiffFile(tif_path) as tif:
            page = tif.pages[0]
            software = page.tags["Software"].value
            artist = json.loads(page.tags["Artist"].value)
        return str(software), artist, tif_path

    def infer_analysis_fs(self, tif_dir: Path) -> float:
        try:
            software, _artist, _tif_path = self.load_first_analysis_tif_metadata(tif_dir)
            fs = self.scanimage_number_from_software(software, "SI.hRoiManager.scanVolumeRate")
            if fs is None:
                fs = self.scanimage_number_from_software(software, "SI.hRoiManager.scanFrameRate")
            if fs is None or not np.isfinite(fs) or fs <= 0:
                raise ValueError("ScanImage metadata does not contain a usable scanVolumeRate or scanFrameRate.")
            self.analysis_fs_label.setText(f"fs: auto {fs:.5g} Hz")
            return float(fs)
        except Exception:
            fallback = float(self.analysis_fs_spin.value())
            self.analysis_fs_label.setText(f"fs: using fallback {fallback:.5g} Hz")
            return fallback

    def infer_analysis_xy_voxel_size_um(self, tif_dir: Path) -> tuple[float, float]:
        try:
            software, artist, _tif_path = self.load_first_analysis_tif_metadata(tif_dir)

            objective_resolution = self.scanimage_number_from_software(
                software,
                "SI.objectiveResolution",
            )
            if objective_resolution is None:
                raise ValueError("ScanImage metadata does not contain SI.objectiveResolution.")

            rois = artist["RoiGroups"]["imagingRoiGroup"]["rois"]
            xy_values = []
            for roi in rois:
                for scanfield in roi.get("scanfields", []):
                    size_xy = scanfield.get("sizeXY")
                    pix_xy = scanfield.get("pixelResolutionXY")
                    if size_xy is None or pix_xy is None:
                        continue
                    if len(size_xy) < 2 or len(pix_xy) < 2:
                        continue
                    x_um = float(size_xy[0]) * objective_resolution / float(pix_xy[0])
                    y_um = float(size_xy[1]) * objective_resolution / float(pix_xy[1])
                    if np.isfinite(y_um) and np.isfinite(x_um) and y_um > 0 and x_um > 0:
                        xy_values.append((y_um, x_um))
            if not xy_values:
                raise ValueError("ScanImage ROI metadata does not contain usable sizeXY/pixelResolutionXY values.")

            xy = np.asarray(xy_values, dtype=float)
            y_um, x_um = np.median(xy, axis=0)
            self.analysis_xy_voxel_label.setText(f"y/x: auto {y_um:.3g}, {x_um:.3g} um")
            return float(y_um), float(x_um)
        except Exception as exc:
            self.analysis_xy_voxel_label.setText("y/x: could not read TIFF metadata")
            raise ValueError(f"Could not infer y/x voxel size from TIFF metadata: {exc}") from exc

    def scanimage_number_from_software(self, software: str, key: str) -> float | None:
        for line in str(software).splitlines():
            if line.startswith(key):
                match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", line)
                if match:
                    return float(match.group(0))
        return None

    def parse_analysis_params(self) -> dict:
        n_ch_tif = int(self.analysis_n_ch_tif_spin.value())
        planes_text = self.analysis_planes_edit.text().strip()
        if planes_text:
            try:
                planes = [int(part.strip()) for part in planes_text.split(",") if part.strip()]
            except ValueError as exc:
                raise ValueError("Planes must be comma-separated integers, for example: 0, 1, 2.") from exc
            if not planes:
                raise ValueError("Enter at least one plane or leave planes blank for all planes.")
        else:
            planes = list(range(n_ch_tif))

        tif_dir = Path(self.analysis_tif_dir_edit.text().strip()).expanduser()
        fs = self.infer_analysis_fs(tif_dir)
        y_um, x_um = self.infer_analysis_xy_voxel_size_um(tif_dir)
        lbm = self.analysis_lbm_checkbox.isChecked()
        cavity_size = 15
        subtract_crosstalk = bool(lbm and len(planes) > cavity_size)

        return {
            "fs": fs,
            "tau": float(self.analysis_tau_spin.value()),
            "voxel_size_um": (
                float(self.analysis_voxel_z_spin.value()),
                y_um,
                x_um,
            ),
            "planes": planes,
            "n_ch_tif": n_ch_tif,
            "lbm": lbm,
            "faced": False,
            "3d_reg": self.analysis_3d_reg_checkbox.isChecked(),
            "gpu_reg": self.analysis_gpu_reg_checkbox.isChecked(),
            "cavity_size": cavity_size,
            "subtract_crosstalk": subtract_crosstalk,
            "fuse_strips": True,
            "neuropil_mask_method": self.analysis_neuropil_mask_method_combo.currentText(),
        }

    def start_analysis(self) -> None:
        if self.analysis_worker is not None and self.analysis_worker.isRunning():
            QMessageBox.information(self, "Analysis running", "An analysis is already running.")
            return

        tif_dir_text = self.analysis_tif_dir_edit.text().strip()
        output_dir_text = self.analysis_output_dir_edit.text().strip()
        tif_dir = Path(tif_dir_text).expanduser()
        output_dir = Path(output_dir_text).expanduser()
        job_id = "gui-analysis"
        if not tif_dir.exists() or not tif_dir.is_dir():
            QMessageBox.warning(self, "Invalid TIFF directory", "Choose a directory containing 2P TIFF files.")
            return
        if not output_dir_text:
            QMessageBox.warning(self, "Invalid saving directory", "Choose a saving directory.")
            return
        try:
            params = self.parse_analysis_params()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid parameters", str(exc))
            return

        self.analysis_log.clear()
        self.analysis_progress.setValue(0)
        self.analysis_stage_label.setText("Starting")
        self.analysis_run_button.setEnabled(False)
        self.analysis_worker = Suite3DAnalysisWorker(
            tif_dir=tif_dir,
            output_dir=output_dir,
            job_id=job_id,
            params=params,
            overwrite=True,
            test_batch_only=self.analysis_test_batch_checkbox.isChecked(),
            parent=self,
        )
        self.analysis_worker.progress_changed.connect(self.on_analysis_progress)
        self.analysis_worker.log_message.connect(self.append_analysis_log)
        self.analysis_worker.finished_with_status.connect(self.on_analysis_finished)
        self.analysis_worker.start()

    def on_analysis_progress(self, value: int, stage: str) -> None:
        self.analysis_progress.setValue(int(value))
        self.analysis_stage_label.setText(stage)

    def append_analysis_log(self, message: str) -> None:
        self.analysis_log.appendPlainText(str(message))

    def on_analysis_finished(self, success: bool, message: str, info_path: str) -> None:
        self.analysis_run_button.setEnabled(True)
        self.analysis_stage_label.setText(message)
        self.append_analysis_log(message)
        if not success:
            self.analysis_progress.setValue(0)
            QMessageBox.critical(self, "Analysis failed", message)
            return

        self.analysis_progress.setValue(100)
        if info_path:
            path = Path(info_path)
            if path.exists():
                self.load_info(path)
                if self.analysis_dialog is not None:
                    self.analysis_dialog.hide()

    def build_curation_panel(self) -> QGroupBox:
        group = QGroupBox("ROI curation")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.correlation_colorbar_figure = Figure(figsize=(4.1, 0.48), dpi=100, constrained_layout=False)
        self.correlation_colorbar_canvas = FigureCanvas(self.correlation_colorbar_figure)
        self.correlation_colorbar_canvas.setStyleSheet(f"background-color: {PANEL_BG};")
        self.correlation_colorbar_canvas.setFixedHeight(48)
        self.correlation_colorbar_canvas.setMinimumWidth(390)
        self.correlation_colorbar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.clear_correlation_colorbar()
        layout.addWidget(self.correlation_colorbar_canvas)

        metric_selector = QGroupBox("Histograms")
        metric_selector_layout = QGridLayout(metric_selector)
        metric_selector_layout.setContentsMargins(6, 6, 6, 6)
        metric_selector_layout.setHorizontalSpacing(10)
        metric_selector_layout.setVerticalSpacing(4)

        for idx, (key, label, decimals) in enumerate(self.curation_metric_definitions()):
            show_by_default = key == "trace_skew"
            self.curation_controls[key] = {
                "active": False,
                "visible": show_by_default,
                "available": True,
                "min": 0.0,
                "max": 0.0,
                "range_min": -1e9,
                "range_max": 1e9,
                "label": label,
                "decimals": decimals,
            }
            checkbox = QCheckBox(label)
            checkbox.setChecked(show_by_default)
            checkbox.stateChanged.connect(self.on_curation_metric_visibility_changed)
            self.curation_metric_checkboxes[key] = checkbox
            metric_selector_layout.addWidget(checkbox, idx // 2, idx % 2)
        layout.addWidget(metric_selector)

        self.curation_figure = Figure(figsize=(4.2, 5.8), dpi=100, facecolor=PANEL_BG)
        self.curation_canvas = FigureCanvas(self.curation_figure)
        self.curation_canvas.setStyleSheet(f"background-color: {PANEL_BG};")
        self.curation_canvas.setAutoFillBackground(True)
        self.curation_canvas.setMinimumWidth(420)
        self.curation_canvas.setMinimumHeight(220)
        self.curation_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.curation_canvas.mpl_connect("button_press_event", self.on_curation_press)
        self.curation_canvas.mpl_connect("motion_notify_event", self.on_curation_motion)
        self.curation_canvas.mpl_connect("button_release_event", self.on_curation_release)
        self.curation_canvas.mpl_connect("figure_leave_event", self.on_curation_leave)
        self.curation_scroll_area = QScrollArea()
        self.curation_scroll_area.setWidgetResizable(True)
        self.curation_scroll_area.setStyleSheet(
            f"QScrollArea {{ background-color: {PANEL_BG}; border: 1px solid #777777; }}"
            f"QScrollArea > QWidget > QWidget {{ background-color: {PANEL_BG}; }}"
        )
        self.curation_scroll_area.setWidget(self.curation_canvas)
        layout.addWidget(self.curation_scroll_area, stretch=1)

        neuropil_reject_row = QHBoxLayout()
        self.reject_neuropil_button = QPushButton("Reject Fneu > F")
        self.reject_neuropil_button.clicked.connect(self.reject_high_neuropil_mean_rois)
        self.reject_neuropil_button.setEnabled(False)
        self.undo_neuropil_reject_button = QPushButton("Undo")
        self.undo_neuropil_reject_button.clicked.connect(self.undo_neuropil_rejection)
        self.undo_neuropil_reject_button.setEnabled(False)
        neuropil_reject_row.addWidget(self.reject_neuropil_button, stretch=1)
        neuropil_reject_row.addWidget(self.undo_neuropil_reject_button)
        layout.addLayout(neuropil_reject_row)

        self.save_iscell_button = QPushButton("Save to iscell.npy")
        self.save_iscell_button.clicked.connect(self.save_iscell)
        self.save_iscell_button.setEnabled(False)
        layout.addWidget(self.save_iscell_button)

        group.setMinimumWidth(450)
        group.setEnabled(False)
        return group

    def curation_metric_definitions(self) -> tuple[tuple[str, str, int], ...]:
        return (
            ("npix", "voxels", 0),
            ("zspan", "z-planes", 0),
            ("peak_val", "peak val", 3),
            ("vox_snr", "vox SNR", 3),
            ("trace_skew", "F skewness", 3),
        )

    def curation_metric_help(self) -> dict[str, str]:
        return {
            "npix": "ROI footprint size in voxels.",
            "zspan": "Number of z-planes touched by the ROI.",
            "peak_val": "Strength of the seed peak in the Suite3D detection map.",
            "vox_snr": "Median per-voxel signal-to-noise inside the ROI; higher values mean the ROI-linked signal stands out more clearly above voxel-level noise.",
            "trace_skew": "Skewness of the ROI F trace computed by the viewer.",
        }

    def on_curation_metric_visibility_changed(self, _state: int) -> None:
        for key, checkbox in self.curation_metric_checkboxes.items():
            controls = self.curation_controls.get(key)
            if controls is None:
                continue
            checked = checkbox.isChecked()
            controls["visible"] = checked
        if self.curation_updating:
            return
        self.update_curation_histograms()

    def visible_curation_metric_definitions(self) -> list[tuple[str, str, int]]:
        visible = []
        for key, label, decimals in self.curation_metric_definitions():
            controls = self.curation_controls.get(key)
            if controls is None:
                continue
            if bool(controls.get("visible", False)) and bool(controls.get("available", False)):
                visible.append((key, label, decimals))
        return visible

    def start_directory(self) -> Path:
        if self.info_path is not None:
            if self.info_path.parent.name == "rois":
                return self.info_path.parent.parent
            return self.info_path.parent
        return Path.home()

    def open_directory_dialog(self) -> None:
        dirname = QFileDialog.getExistingDirectory(
            self,
            "Open Suite3D run folder",
            str(self.start_directory()),
        )
        if dirname:
            self.load_info(Path(dirname))

    def open_file_dialog(self) -> None:
        start_dir = str(self.start_directory())
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Suite3D info.npy",
            start_dir,
            "NumPy files (*.npy);;All files (*.*)",
        )
        if filename:
            self.load_info(Path(filename))

    def resolve_info_path(self, path: Path) -> Path:
        if path.is_dir():
            candidates = (
                path / "rois" / "info.npy",
                path / "info.npy",
            )
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(
                f"Could not find rois\\info.npy or info.npy inside:\n{path}"
            )
        return path

    def load_info(self, path: Path) -> None:
        try:
            path = self.resolve_info_path(path)
            loaded = np.load(path, allow_pickle=True)
            info = loaded.item()
            if not isinstance(info, dict):
                raise ValueError("This file does not contain a Suite3D info dictionary.")
        except Exception as exc:
            QMessageBox.critical(self, "Could not load info.npy", str(exc))
            return

        available = []
        for key in DISPLAY_KEYS:
            value = info.get(key)
            if isinstance(value, np.ndarray) and value.ndim == 3:
                available.append(key)

        if not available:
            QMessageBox.critical(
                self,
                "No displayable images",
                "Could not find any 3D arrays named max_img, mean_img, vmap, or vmap_raw.",
            )
            return

        self.info_path = path
        self.info = info
        self.file_label.setText(str(path))
        self.load_roi_files(path.parent)
        self.load_recording_files(path.parent)

        self.data_combo.blockSignals(True)
        self.data_combo.clear()
        self.data_combo.addItems(available)
        self.data_combo.addItem(BLACK_DISPLAY_KEY)
        if self.recording_files:
            self.data_combo.addItem(RECORDING_DISPLAY_KEY)
        if "max_img" in available:
            self.data_combo.setCurrentText("max_img")
        self.data_combo.blockSignals(False)

        self.update_array_selection()
        self.update_image()

    def load_roi_files(self, directory: Path) -> None:
        self.stats = None
        self.iscell = None
        self.roi_colors = None
        self.clear_correlation_colors(update_view=False)
        self.roi_metrics = {}
        if hasattr(self, "curation_group"):
            self.curation_group.setEnabled(False)
            self.save_iscell_button.setEnabled(False)
            self.reject_neuropil_button.setEnabled(False)
            self.undo_neuropil_reject_button.setEnabled(False)
        self.current_id_maps = {}
        self.image_axes_panels = {}
        self.image_axes_roles = {}
        self.overlay_cache = {}
        self.selected_roi_idx = None
        self.selected_panel_accepted = True
        self.selected_mask_kind = "roi"
        self.update_show_3d_button_text()
        self.curation_undo_stack = []
        self.manual_curation_overrides = {}
        self.neuropil_rejected_roi_indices = set()
        self.clear_selection_rectangle()
        self.traces = {}
        self.trace_paths = {}
        self.trace_roi_axes = {}

        stats_path = directory / "stats.npy"
        if not stats_path.exists():
            self.masks_checkbox.setChecked(False)
            self.masks_checkbox.setEnabled(False)
            self.secondary_panel_combo.setEnabled(False)
            self.roi_color_combo.setCurrentText("Random")
            self.roi_color_combo.setEnabled(False)
            self.clear_trace_plot("F trace: no stats.npy found")
            return

        try:
            self.stats = np.load(stats_path, allow_pickle=True)
        except Exception as exc:
            self.masks_checkbox.setChecked(False)
            self.masks_checkbox.setEnabled(False)
            self.secondary_panel_combo.setEnabled(False)
            self.roi_color_combo.setCurrentText("Random")
            self.roi_color_combo.setEnabled(False)
            self.clear_trace_plot("F trace: could not load stats.npy")
            return

        iscell_path = directory / "iscell.npy"
        if iscell_path.exists():
            try:
                self.iscell = np.load(iscell_path, allow_pickle=True)
            except Exception:
                self.iscell = None

        rng = np.random.default_rng(12345)
        self.roi_colors = rng.random((len(self.stats), 3), dtype=np.float32)
        self.roi_colors = 0.25 + 0.75 * self.roi_colors

        self.masks_checkbox.setEnabled(True)
        self.secondary_panel_combo.setEnabled(True)
        self.roi_color_combo.setEnabled(True)
        self.load_trace_file(directory)
        self.initialize_curation_controls()

    def initialize_curation_controls(self) -> None:
        if self.stats is None:
            return

        self.roi_metrics = self.compute_roi_metrics()
        self.ensure_iscell_array()
        self.curation_updating = True
        try:
            for key, _label, _decimals in self.curation_metric_definitions():
                controls = self.curation_controls.get(key)
                checkbox = self.curation_metric_checkboxes.get(key)
                values = self.roi_metrics.get(key)
                if controls is None or values is None:
                    continue
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    controls["available"] = False
                    controls["visible"] = False
                    controls["active"] = False
                    if checkbox is not None:
                        checkbox.blockSignals(True)
                        checkbox.setChecked(False)
                        checkbox.setEnabled(False)
                        checkbox.blockSignals(False)
                    continue
                min_value = float(np.nanmin(finite))
                max_value = float(np.nanmax(finite))
                margin = max((max_value - min_value) * 0.05, 1.0)
                controls["range_min"] = min_value - margin
                controls["range_max"] = max_value + margin
                controls["min"] = min_value
                controls["max"] = max_value
                controls["available"] = True
                if checkbox is not None:
                    checkbox.blockSignals(True)
                    checkbox.setEnabled(True)
                    controls["visible"] = checkbox.isChecked()
                    checkbox.blockSignals(False)
                else:
                    controls["visible"] = True
        finally:
            self.curation_updating = False

        self.curation_group.setEnabled(True)
        self.save_iscell_button.setEnabled(True)
        self.update_curation_histograms()

    def compute_roi_metrics(self) -> dict[str, np.ndarray]:
        if self.stats is None:
            return {}

        n_rois = len(self.stats)
        metrics = {
            "npix": np.full(n_rois, np.nan, dtype=float),
            "zspan": np.full(n_rois, np.nan, dtype=float),
            "peak_val": np.full(n_rois, np.nan, dtype=float),
            "vox_snr": np.full(n_rois, np.nan, dtype=float),
            "trace_skew": np.full(n_rois, np.nan, dtype=float),
        }
        metrics["trace_skew"][:] = self.stored_trace_skewness(n_rois)
        missing_skew = ~np.isfinite(metrics["trace_skew"])
        if np.any(missing_skew):
            trace_skew = self.compute_f_trace_skewness(n_rois)
            if trace_skew is not None:
                metrics["trace_skew"][missing_skew] = trace_skew[missing_skew]

        for roi_idx, stat in enumerate(self.stats):
            coords = stat.get("coords")
            lam = stat.get("lam")
            if lam is not None:
                metrics["npix"][roi_idx] = float(np.asarray(lam).size)
            elif coords is not None and len(coords) == 3:
                metrics["npix"][roi_idx] = float(np.asarray(coords[0]).size)

            if coords is not None and len(coords) == 3:
                metrics["zspan"][roi_idx] = float(np.unique(np.asarray(coords[0])).size)

            value = stat.get("peak_val")
            if value is not None:
                array = np.asarray(value)
                if array.size == 1:
                    metrics["peak_val"][roi_idx] = float(array)

            vox_snrs = stat.get("vox_snrs")
            if vox_snrs is not None:
                finite = np.asarray(vox_snrs, dtype=float)
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    metrics["vox_snr"][roi_idx] = float(np.median(finite))
        return metrics

    def stored_trace_skewness(self, n_rois: int) -> np.ndarray:
        skew = np.full(n_rois, np.nan, dtype=float)
        if self.stats is None:
            return skew
        for roi_idx, stat in enumerate(self.stats[:n_rois]):
            for key in SKEW_STAT_KEYS:
                value = stat.get(key)
                if value is None:
                    continue
                array = np.asarray(value, dtype=float)
                if array.size == 1:
                    skew[roi_idx] = float(array)
                    break
        return skew

    def compute_f_trace_skewness(self, n_rois: int) -> np.ndarray | None:
        trace_array = self.traces.get("F")
        roi_axis = self.trace_roi_axes.get("F")
        if trace_array is None or roi_axis is None:
            return None

        if trace_array.ndim != 2:
            return None

        if roi_axis == 0:
            if trace_array.shape[0] != n_rois:
                return None
        elif roi_axis == 1:
            if trace_array.shape[1] != n_rois:
                return None
        else:
            return None

        skew = np.full(n_rois, np.nan, dtype=float)
        for start in range(0, n_rois, TRACE_SKEW_CHUNK_ROIS):
            end = min(start + TRACE_SKEW_CHUNK_ROIS, n_rois)
            if roi_axis == 0:
                traces = np.asarray(trace_array[start:end], dtype=np.float32)
            else:
                traces = np.asarray(trace_array[:, start:end], dtype=np.float32).T
            skew[start:end] = self.trace_chunk_skewness(traces)
        return skew

    @staticmethod
    def trace_chunk_skewness(traces: np.ndarray) -> np.ndarray:
        n_chunk = traces.shape[0]
        finite = np.isfinite(traces)
        counts = finite.sum(axis=1)
        safe = finite & (counts[:, None] >= 3)
        traces_zeroed = np.where(safe, traces, 0.0)
        sums = traces_zeroed.sum(axis=1)
        means = np.divide(sums, counts, out=np.full(n_chunk, np.nan, dtype=np.float32), where=counts > 0)

        centered = np.where(safe, traces - means[:, None], 0.0)
        second = np.divide(
            np.sum(centered * centered, axis=1),
            counts,
            out=np.full(n_chunk, np.nan, dtype=np.float32),
            where=counts >= 3,
        )
        std = np.sqrt(second)
        third = np.divide(
            np.sum(centered * centered * centered, axis=1),
            counts,
            out=np.full(n_chunk, np.nan, dtype=np.float32),
            where=counts >= 3,
        )
        skew = np.divide(
            third,
            std * std * std,
            out=np.full(n_chunk, np.nan, dtype=np.float32),
            where=(counts >= 3) & np.isfinite(std) & (std > 0),
        )
        return skew.astype(float, copy=False)

    def ensure_iscell_array(self) -> None:
        if self.stats is None:
            return
        n_rois = len(self.stats)
        if self.iscell is None or self.iscell.shape[0] != n_rois or self.iscell.ndim != 2:
            self.iscell = np.zeros((n_rois, 2), dtype=np.int64)
            self.iscell[:, 0] = 1
            self.iscell[:, 1] = 1

    def on_curation_changed(self, *_args) -> None:
        if self.curation_updating or self.stats is None:
            return
        self.apply_curation_filters()

    def apply_curation_filters(self, push_undo: bool = True) -> None:
        self.ensure_iscell_array()
        if self.iscell is None or not self.roi_metrics:
            return
        if push_undo:
            self.push_curation_undo_state()

        accepted = np.ones(self.iscell.shape[0], dtype=bool)
        for key, _label, _decimals in self.curation_metric_definitions():
            controls = self.curation_controls.get(key)
            values = self.roi_metrics.get(key)
            if (
                controls is None
                or values is None
                or not bool(controls.get("available", False))
                or not bool(controls.get("active", False))
            ):
                continue
            low = float(controls["min"])
            high = float(controls["max"])
            if low > high:
                low, high = high, low
            accepted &= np.isfinite(values) & (values >= low) & (values <= high)

        self.manual_curation_overrides = {}
        for roi_idx in self.neuropil_rejected_roi_indices:
            if 0 <= roi_idx < accepted.shape[0]:
                accepted[roi_idx] = False
        self.iscell[:, 0] = accepted.astype(self.iscell.dtype)
        self.overlay_cache = {}
        self.update_neuropil_rejection_undo_button()
        self.update_curation_histograms()
        self.update_image(preserve_view=True)

    def update_curation_histograms(self) -> None:
        if not hasattr(self, "curation_figure"):
            return
        self.curation_figure.clear()
        self.curation_figure.patch.set_facecolor(PANEL_BG)
        self.curation_axes = {}
        self.curation_threshold_lines = {}
        self.curation_title_artists = {}
        self.curation_tooltip_key = None

        if self.stats is None or not self.roi_metrics:
            ax = self.curation_figure.add_subplot(111)
            ax.set_facecolor(PLOT_BG)
            ax.text(0.5, 0.5, "No ROI masks loaded", ha="center", va="center", color=TEXT_FG)
            ax.set_axis_off()
            self.curation_canvas.draw_idle()
            return

        metric_defs = self.visible_curation_metric_definitions()
        if not metric_defs:
            self.curation_canvas.setMinimumHeight(180)
            self.curation_figure.set_size_inches(4.2, 1.8, forward=True)
            ax = self.curation_figure.add_subplot(111)
            ax.set_facecolor(PLOT_BG)
            ax.text(0.5, 0.5, "No histograms selected", ha="center", va="center", color=TEXT_FG)
            ax.set_axis_off()
            self.curation_canvas.draw_idle()
            return

        panel_height = 1.65
        figure_height = max(2.0, panel_height * len(metric_defs))
        self.curation_figure.set_size_inches(4.2, figure_height, forward=True)
        self.curation_canvas.setMinimumHeight(max(220, int(155 * len(metric_defs))))
        self.curation_canvas.updateGeometry()
        axes = np.atleast_1d(self.curation_figure.subplots(len(metric_defs), 1))
        for ax, (key, label, _decimals) in zip(axes, metric_defs):
            values = self.roi_metrics.get(key)
            self.curation_axes[ax] = key
            ax.set_facecolor(PLOT_BG)
            if values is None:
                ax.set_axis_off()
                continue
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                ax.text(0.5, 0.5, "No values", ha="center", va="center", color=TEXT_FG)
                ax.set_axis_off()
                continue

            ax.hist(finite, bins=45, color="#b8b8b8", edgecolor="#777777", linewidth=0.3)
            controls = self.curation_controls.get(key)
            if controls is not None and bool(controls.get("available", False)):
                low = float(controls["min"])
                high = float(controls["max"])
                if low > high:
                    low, high = high, low
                line_color = "#d6d14a" if bool(controls.get("active", False)) else "#8f8a3a"
                line_style = "-" if bool(controls.get("active", False)) else "--"
                low_line = ax.axvline(low, color=line_color, linestyle=line_style, linewidth=1.8, picker=6)
                high_line = ax.axvline(high, color=line_color, linestyle=line_style, linewidth=1.8, picker=6)
                self.curation_threshold_lines[(key, "min")] = low_line
                self.curation_threshold_lines[(key, "max")] = high_line
            title = ax.set_title(f"{label}: {finite.min():.4g} - {finite.max():.4g}", color=TEXT_FG, fontsize=8)
            self.curation_title_artists[title] = key
            ax.tick_params(colors=TEXT_FG, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(GRID_FG)

        if len(metric_defs) == 1:
            self.curation_figure.subplots_adjust(left=0.13, right=0.98, top=0.82, bottom=0.22)
        else:
            self.curation_figure.subplots_adjust(left=0.13, right=0.98, top=0.94, bottom=0.06, hspace=0.85)
        self.curation_canvas.draw_idle()

    def on_curation_press(self, event) -> None:
        QToolTip.hideText()
        self.curation_tooltip_key = None
        if not self.is_left_mouse_button(event.button) or event.inaxes not in self.curation_axes:
            return
        if event.xdata is None:
            return
        key = self.curation_axes[event.inaxes]
        controls = self.curation_controls.get(key)
        if controls is None or not bool(controls.get("available", False)):
            return

        candidates = []
        for bound in ("min", "max"):
            line = self.curation_threshold_lines.get((key, bound))
            if line is None:
                continue
            x_value = float(line.get_xdata()[0])
            x_pixel = event.inaxes.transData.transform((x_value, 0))[0]
            candidates.append((abs(float(event.x) - float(x_pixel)), bound))
        if not candidates:
            return
        distance, bound = min(candidates, key=lambda item: item[0])
        if distance > 18:
            return
        self.curation_drag = {"key": key, "bound": bound, "axes": event.inaxes}
        self.set_curation_bound_from_histogram(key, bound, float(event.xdata), apply_filters=False)

    def on_curation_motion(self, event) -> None:
        if self.curation_drag is None or event.xdata is None:
            self.update_curation_tooltip(event)
            return
        QToolTip.hideText()
        self.curation_tooltip_key = None
        self.set_curation_bound_from_histogram(
            str(self.curation_drag["key"]),
            str(self.curation_drag["bound"]),
            float(event.xdata),
            apply_filters=False,
        )

    def on_curation_release(self, event) -> None:
        if self.curation_drag is None:
            return
        QToolTip.hideText()
        self.curation_tooltip_key = None
        if event.xdata is not None:
            self.set_curation_bound_from_histogram(
                str(self.curation_drag["key"]),
                str(self.curation_drag["bound"]),
                float(event.xdata),
                apply_filters=False,
            )
        self.curation_drag = None
        self.apply_curation_filters()

    def set_curation_bound_from_histogram(
        self,
        key: str,
        bound: str,
        value: float,
        apply_filters: bool,
    ) -> None:
        controls = self.curation_controls.get(key)
        if controls is None or bound not in ("min", "max"):
            return
        value = min(max(float(value), float(controls["range_min"])), float(controls["range_max"]))
        self.curation_updating = True
        try:
            controls[bound] = value
            controls["active"] = True
        finally:
            self.curation_updating = False
        line = self.curation_threshold_lines.get((key, bound))
        if line is not None:
            line.set_xdata([value, value])
            self.curation_canvas.draw_idle()
        if apply_filters:
            self.apply_curation_filters()

    def trace_mean_by_roi(self, trace_key: str) -> np.ndarray | None:
        trace_array = self.traces.get(trace_key)
        roi_axis = self.trace_roi_axes.get(trace_key)
        if trace_array is None or roi_axis is None:
            return None
        with np.errstate(invalid="ignore"):
            if roi_axis == 0:
                return np.asarray(np.nanmean(trace_array, axis=1), dtype=float)
            return np.asarray(np.nanmean(trace_array, axis=0), dtype=float)

    def on_roi_color_mode_changed(self, mode: str) -> None:
        if mode == "Random":
            self.clear_correlation_colors(update_view=True)
            return
        if not self.masks_checkbox.isChecked():
            self.masks_checkbox.setChecked(True)
        if mode == "Skewness":
            if not self.update_metric_roi_colors("trace_skew", "Skewness", "F skewness", "viridis"):
                self.reset_roi_color_combo_to_random()
            return
        if mode == "Voxel count":
            if not self.update_metric_roi_colors("npix", "Voxel count", "voxels", "plasma"):
                self.reset_roi_color_combo_to_random()
            return
        if mode == "Peak value":
            if not self.update_metric_roi_colors("peak_val", "Peak value", "peak value", "inferno"):
                self.reset_roi_color_combo_to_random()
            return
        if mode == "Voxel SNR":
            if not self.update_metric_roi_colors("vox_snr", "Voxel SNR", "voxel SNR", "magma"):
                self.reset_roi_color_combo_to_random()
            return
        if mode == "Correlation":
            if self.selected_roi_idx is None:
                self.clear_correlation_colors(update_view=False)
                self.status_label.setText("Correlation colors: select an ROI first.")
                self.update_image(preserve_view=True)
                return
            if not self.update_correlation_colors(self.selected_roi_idx):
                self.reset_roi_color_combo_to_random()
            return
        self.clear_correlation_colors(update_view=True)

    def reset_roi_color_combo_to_random(self) -> None:
        if hasattr(self, "roi_color_combo"):
            self.roi_color_combo.blockSignals(True)
            self.roi_color_combo.setCurrentText("Random")
            self.roi_color_combo.blockSignals(False)
        self.clear_correlation_colors(update_view=True)

    def update_metric_roi_colors(self, metric_key: str, mode: str, label: str, cmap_name: str) -> bool:
        values = self.roi_metrics.get(metric_key) if self.roi_metrics else None
        if values is None and self.stats is not None:
            self.roi_metrics = self.compute_roi_metrics()
            values = self.roi_metrics.get(metric_key)
        if values is None:
            QMessageBox.information(self, f"Missing {label}", f"No {label} values are available.")
            return False

        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            QMessageBox.information(self, f"Missing {label}", f"No finite {label} values are available.")
            return False

        limits = self.correlation_color_limits_for_values(values, seed_roi_idx=-1)
        colors = self.colors_for_roi_values(values, limits, cmap_name=cmap_name, center_zero=False)
        self.set_roi_value_colors(
            values,
            colors,
            limits,
            mode=mode,
            label=label,
            seed_roi_idx=None,
            center_zero=False,
            cmap_name=cmap_name,
        )
        self.status_label.setText(f"{label} colors: range {limits[0]:.3f} to {limits[1]:.3f}.")
        self.update_image(preserve_view=True)
        return True

    def update_skewness_colors(self) -> bool:
        return self.update_metric_roi_colors("trace_skew", "Skewness", "F skewness", "viridis")

    def clear_correlation_colors(self, update_view: bool) -> None:
        self.correlation_colors = None
        self.correlation_values = None
        self.correlation_seed_roi_idx = None
        self.correlation_color_limits = None
        self.roi_color_overlay_mode = None
        self.roi_colorbar_label = None
        self.roi_colorbar_cmap_name = "coolwarm"
        self.roi_colorbar_center_zero = False
        self.overlay_cache = {}
        self.clear_correlation_colorbar()
        if update_view and hasattr(self, "canvas"):
            self.update_image(preserve_view=True)

    def correlation_trace_matrix(self) -> np.ndarray | None:
        trace_array = self.traces.get("F")
        roi_axis = self.trace_roi_axes.get("F")
        if trace_array is None or roi_axis is None or self.stats is None:
            return None

        traces = np.asarray(trace_array, dtype=np.float32)
        if roi_axis == 1:
            traces = traces.T
        if traces.ndim != 2 or traces.shape[0] != len(self.stats):
            return None
        return traces

    def update_correlation_colors(self, seed_roi_idx: int) -> bool:
        traces = self.correlation_trace_matrix()
        if traces is None:
            QMessageBox.information(self, "Missing traces", "F.npy is required for correlation colors.")
            return False
        if not (0 <= seed_roi_idx < traces.shape[0]):
            return False

        values = self.compute_trace_correlations(traces, seed_roi_idx)
        if values is None:
            QMessageBox.information(self, "Correlation failed", f"Could not compute correlations for ROI {seed_roi_idx}.")
            return False

        limits = self.correlation_color_limits_for_values(values, seed_roi_idx)
        colors = self.colors_for_roi_values(values, limits, cmap_name="coolwarm", center_zero=True)
        colors[seed_roi_idx] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.set_roi_value_colors(
            values,
            colors,
            limits,
            mode="Correlation",
            label=f"corr ROI {seed_roi_idx}",
            seed_roi_idx=seed_roi_idx,
            center_zero=True,
            cmap_name="coolwarm",
        )
        finite = values[np.isfinite(values)]
        if finite.size:
            self.status_label.setText(
                f"Correlation colors: seed ROI {seed_roi_idx}, "
                f"color range {limits[0]:.3f} to {limits[1]:.3f}."
            )
        else:
            self.status_label.setText(f"Correlation colors: seed ROI {seed_roi_idx}.")
        self.update_image(preserve_view=True)
        return True

    def colors_for_roi_values(
        self,
        values: np.ndarray,
        limits: tuple[float, float],
        cmap_name: str,
        center_zero: bool,
    ) -> np.ndarray:
        try:
            from matplotlib import cm
            from matplotlib.colors import Normalize, TwoSlopeNorm

            vmin, vmax = limits
            if center_zero and vmin < 0.0 < vmax:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)
            colors = cm.get_cmap(cmap_name)(norm(values))[:, :3].astype(np.float32)
        except Exception:
            vmin, vmax = limits
            span = max(vmax - vmin, 1e-9)
            scaled = np.clip((values - vmin) / span, 0.0, 1.0)
            colors = np.zeros((values.size, 3), dtype=np.float32)
            colors[:, 0] = scaled
            colors[:, 2] = 1.0 - scaled

        colors[~np.isfinite(values)] = np.array([0.35, 0.35, 0.35], dtype=np.float32)
        return colors

    def set_roi_value_colors(
        self,
        values: np.ndarray,
        colors: np.ndarray,
        limits: tuple[float, float],
        mode: str,
        label: str,
        seed_roi_idx: int | None,
        center_zero: bool,
        cmap_name: str,
    ) -> None:
        self.correlation_values = np.asarray(values, dtype=float)
        self.correlation_colors = colors
        self.correlation_seed_roi_idx = seed_roi_idx
        self.correlation_color_limits = limits
        self.roi_color_overlay_mode = mode
        self.roi_colorbar_label = label
        self.roi_colorbar_cmap_name = cmap_name
        self.roi_colorbar_center_zero = bool(center_zero)
        self.update_correlation_colorbar()
        self.overlay_cache = {}

    def correlation_color_limits_for_values(self, values: np.ndarray, seed_roi_idx: int) -> tuple[float, float]:
        finite_mask = np.isfinite(values)
        if 0 <= seed_roi_idx < finite_mask.size:
            finite_mask[seed_roi_idx] = False
        finite = values[finite_mask]
        if finite.size == 0:
            finite = values[np.isfinite(values)]
        if finite.size == 0:
            return -1.0, 1.0

        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return -1.0, 1.0
        if vmax <= vmin:
            margin = max(abs(vmin) * 0.05, 0.05)
            return vmin - margin, vmax + margin
        return vmin, vmax

    def compute_trace_correlations(self, traces: np.ndarray, seed_roi_idx: int) -> np.ndarray | None:
        with np.errstate(invalid="ignore"):
            traces = np.asarray(traces, dtype=np.float32)
            seed = traces[seed_roi_idx].astype(np.float32, copy=True)
            valid_seed = np.isfinite(seed)
            if valid_seed.sum() < 2:
                return None

            traces = traces[:, valid_seed]
            seed = seed[valid_seed]
            valid_trace_counts = np.isfinite(traces).sum(axis=1)
            traces = np.nan_to_num(traces, nan=0.0, posinf=0.0, neginf=0.0)

            seed = seed - float(seed.mean())
            seed_norm = float(np.sqrt(np.dot(seed, seed)))
            if seed_norm <= 0:
                return None

            traces = traces - traces.mean(axis=1, keepdims=True)
            trace_norms = np.sqrt(np.sum(traces * traces, axis=1))
            denom = trace_norms * seed_norm
            values = np.divide(
                traces @ seed,
                denom,
                out=np.full(traces.shape[0], np.nan, dtype=np.float32),
                where=(denom > 0) & (valid_trace_counts >= 2),
            )
        return np.clip(values.astype(float, copy=False), -1.0, 1.0)

    def reject_high_neuropil_mean_rois(self) -> None:
        self.ensure_iscell_array()
        if self.iscell is None:
            return
        f_mean = self.trace_mean_by_roi("F")
        fneu_mean = self.trace_mean_by_roi("Fneu")
        if f_mean is None or fneu_mean is None:
            QMessageBox.information(self, "Missing traces", "F.npy and Fneu.npy are required for this check.")
            return

        n_rois = min(f_mean.size, fneu_mean.size, self.iscell.shape[0])
        reject = np.isfinite(f_mean[:n_rois]) & np.isfinite(fneu_mean[:n_rois]) & (fneu_mean[:n_rois] > f_mean[:n_rois])
        roi_indices = np.flatnonzero(reject).astype(np.int64)
        if roi_indices.size == 0:
            self.status_label.setText("No ROIs found with mean Fneu above mean F.")
            return

        reply = QMessageBox.question(
            self,
            "Reject ROIs",
            f"Move {roi_indices.size} ROIs with mean Fneu > mean F to non-accepted?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.push_curation_undo_state()
        self.neuropil_rejected_roi_indices.update(int(roi_idx) for roi_idx in roi_indices)
        self.apply_curation_filters(push_undo=False)
        self.update_neuropil_rejection_undo_button()
        self.status_label.setText(f"Marked {roi_indices.size} ROIs as non-accepted because mean Fneu > mean F.")

    def undo_neuropil_rejection(self) -> None:
        if not self.neuropil_rejected_roi_indices:
            return
        n_rejected = len(self.neuropil_rejected_roi_indices)
        self.push_curation_undo_state()
        self.neuropil_rejected_roi_indices = set()
        self.apply_curation_filters(push_undo=False)
        self.update_neuropil_rejection_undo_button()
        self.status_label.setText(f"Undid Fneu > F rejection for {n_rejected} ROIs.")

    def update_neuropil_rejection_undo_button(self) -> None:
        if hasattr(self, "undo_neuropil_reject_button"):
            self.undo_neuropil_reject_button.setEnabled(bool(self.neuropil_rejected_roi_indices))

    def navigate_selected_roi(self, step: int) -> None:
        if self.iscell is None or self.iscell.size == 0:
            return
        accepted_panel = bool(self.selected_panel_accepted)
        accepted = self.iscell[:, 0] > 0
        roi_pool = np.flatnonzero(accepted if accepted_panel else ~accepted)
        if roi_pool.size == 0:
            return

        if self.selected_roi_idx is None:
            pool_pos = 0 if step >= 0 else roi_pool.size - 1
        else:
            matches = np.flatnonzero(roi_pool == self.selected_roi_idx)
            if matches.size:
                pool_pos = (int(matches[0]) + step) % roi_pool.size
            else:
                insert_pos = int(np.searchsorted(roi_pool, self.selected_roi_idx))
                pool_pos = insert_pos % roi_pool.size if step >= 0 else (insert_pos - 1) % roi_pool.size

        mask_kind = self.selected_mask_kind if accepted_panel else "roi"
        self.select_roi_by_index(int(roi_pool[pool_pos]), accepted_panel, mask_kind=mask_kind)

    def select_roi_by_index(
        self,
        roi_idx: int,
        accepted_panel: bool | None = None,
        mask_kind: str = "roi",
    ) -> None:
        if self.stats is None or not (0 <= roi_idx < len(self.stats)):
            return
        self.selected_roi_idx = roi_idx
        if accepted_panel is None:
            accepted_panel = bool(self.iscell is None or self.iscell[roi_idx, 0] > 0)
        self.selected_panel_accepted = bool(accepted_panel)
        if mask_kind == "neuropil" and self.roi_mask_coords(self.stats[roi_idx], "neuropil") is not None:
            self.selected_mask_kind = "neuropil"
        else:
            self.selected_mask_kind = "roi"
        self.update_show_3d_button_text()
        self.update_trace_plot(roi_idx)
        if self.roi_color_combo.currentText() == "Correlation":
            self.update_correlation_colors(roi_idx)
            return
        self.update_image(preserve_view=True)

    def update_show_3d_button_text(self) -> None:
        if not hasattr(self, "show_3d_button"):
            return
        if self.selected_mask_kind == "neuropil":
            self.show_3d_button.setText("Show selected neuropil in 3D")
        else:
            self.show_3d_button.setText("Show selected ROI in 3D")

    def update_curation_tooltip(self, event) -> None:
        if event is None or event.x is None or event.y is None:
            self.hide_curation_tooltip()
            return
        renderer = self.curation_canvas.get_renderer()
        help_text = self.curation_metric_help()
        for title, key in self.curation_title_artists.items():
            try:
                if title.get_window_extent(renderer=renderer).expanded(1.08, 1.6).contains(event.x, event.y):
                    if self.curation_tooltip_key != key:
                        self.curation_tooltip_key = key
                        gui_event = getattr(event, "guiEvent", None)
                        if gui_event is not None:
                            pos = self.curation_canvas.mapToGlobal(gui_event.pos())
                        else:
                            pos = self.curation_canvas.mapToGlobal(self.curation_canvas.rect().center())
                        QToolTip.showText(pos, help_text.get(key, ""), self.curation_canvas)
                    return
            except Exception:
                continue
        self.hide_curation_tooltip()

    def hide_curation_tooltip(self) -> None:
        if self.curation_tooltip_key is not None:
            self.curation_tooltip_key = None
            QToolTip.hideText()

    def on_curation_leave(self, _event) -> None:
        self.hide_curation_tooltip()

    def save_iscell(self) -> None:
        if self.info_path is None or self.iscell is None:
            QMessageBox.information(self, "Cannot save", "No iscell data is loaded.")
            return
        path = self.info_path.parent / "iscell.npy"
        try:
            np.save(path, self.iscell)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save {path}:\n{exc}")
            return
        accepted_count = int((self.iscell[:, 0] > 0).sum())
        QMessageBox.information(
            self,
            "Saved",
            f"Saved {path}\nAccepted: {accepted_count}\nNon-accepted: {len(self.iscell) - accepted_count}",
        )

    def load_trace_file(self, directory: Path) -> None:
        self.traces = {}
        self.trace_paths = {}
        self.trace_roi_axes = {}

        if self.stats is None:
            self.reject_neuropil_button.setEnabled(False)
            self.undo_neuropil_reject_button.setEnabled(False)
            self.clear_trace_plot("F trace: no ROI masks loaded")
            return

        n_rois = len(self.stats)
        search_dirs = [directory, directory.parent, directory.parent / "rois"]
        missing = []
        for trace_key, filename in TRACE_FILES.items():
            loaded = False
            for search_dir in search_dirs:
                path = search_dir / filename
                if not path.exists():
                    continue
                try:
                    trace_array = np.load(path, mmap_mode="r", allow_pickle=False)
                except Exception:
                    continue
                if trace_array.ndim != 2:
                    continue
                if trace_array.shape[0] == n_rois:
                    self.traces[trace_key] = trace_array
                    self.trace_paths[trace_key] = path
                    self.trace_roi_axes[trace_key] = 0
                    loaded = True
                    break
                if trace_array.shape[1] == n_rois:
                    self.traces[trace_key] = trace_array
                    self.trace_paths[trace_key] = path
                    self.trace_roi_axes[trace_key] = 1
                    loaded = True
                    break
            if not loaded:
                missing.append(filename)

        if self.traces:
            loaded_names = ", ".join(path.name for path in self.trace_paths.values())
            self.clear_trace_plot(f"Traces loaded: {loaded_names}; select an ROI")
        else:
            self.clear_trace_plot("F trace: no F.npy/Fneu.npy/spks.npy files found next to this Suite3D output")
        have_f_and_fneu = "F" in self.traces and "Fneu" in self.traces
        self.reject_neuropil_button.setEnabled(have_f_and_fneu)
        self.update_neuropil_rejection_undo_button()
        has_f_trace = "F" in self.traces
        if not has_f_trace and self.roi_color_combo.currentText() == "Correlation":
            self.clear_correlation_colors(update_view=False)
            self.roi_color_combo.setCurrentText("Random")

    def load_recording_files(self, rois_dir: Path) -> None:
        self.recording_files = []
        self.recording_frame_counts = []
        self.recording_frame_starts = None
        self.recording_shape = None
        self.recording_chunk_cache = None
        self.motion_shifts = None
        self.motion_shift_paths = []
        self.motion_checkbox.blockSignals(True)
        self.motion_checkbox.setChecked(False)
        self.motion_checkbox.setEnabled(False)
        self.motion_checkbox.blockSignals(False)

        recording_dir = rois_dir.parent / "registered_fused_data"
        files = sorted(recording_dir.glob("fused_reg_data*.npy"))
        if not files:
            self.frame_slider.setEnabled(False)
            self.stop_frame_play()
            self.update_play_button_states()
            self.frame_label.setText("Frame: no recording")
            self.status_label.setText(f"Registered movie not found in {recording_dir}")
            return

        for path in files:
            try:
                arr = np.load(path, mmap_mode="r", allow_pickle=False)
            except Exception:
                continue
            if arr.ndim != 4:
                continue
            nz, nt, ny, nx = arr.shape
            if self.recording_shape is None:
                self.recording_shape = (int(nz), int(ny), int(nx))
            elif self.recording_shape != (int(nz), int(ny), int(nx)):
                continue
            self.recording_files.append(path)
            self.recording_frame_counts.append(int(nt))

        if not self.recording_files:
            self.frame_slider.setEnabled(False)
            self.stop_frame_play()
            self.update_play_button_states()
            self.frame_label.setText("Frame: no recording")
            self.status_label.setText(f"No readable fused_reg_data*.npy files found in {recording_dir}")
            return

        starts = np.concatenate([[0], np.cumsum(self.recording_frame_counts)])
        self.recording_frame_starts = starts.astype(int)
        total_frames = int(starts[-1])
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(0, total_frames - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.frame_slider.setEnabled(True)
        self.frame_label.setText(f"Frame: 0 / {total_frames - 1}")
        self.load_motion_correction_shifts(recording_dir)
        self.update_play_button_states()

    def load_motion_correction_shifts(self, recording_dir: Path) -> None:
        self.motion_shifts = None
        self.motion_shift_paths = []
        offset_files = sorted(recording_dir.glob("offsets*.npy"))
        if not offset_files:
            self.motion_checkbox.setEnabled(False)
            return

        chunks = []
        paths = []
        for path in offset_files:
            try:
                offsets = np.load(path, allow_pickle=True).item()
                shifts = np.asarray(offsets["sub_pixel_shifts"], dtype=float)
            except Exception:
                continue
            if shifts.ndim != 2 or shifts.shape[1] != 3:
                continue
            chunks.append(shifts)
            paths.append(path)

        if not chunks:
            self.motion_checkbox.setEnabled(False)
            return

        self.motion_shifts = np.concatenate(chunks, axis=0)
        self.motion_shift_paths = paths
        self.motion_checkbox.setEnabled(True)

    def is_recording_selected(self) -> bool:
        return self.data_combo.currentText() == RECORDING_DISPLAY_KEY

    def is_black_background_selected(self) -> bool:
        return self.data_combo.currentText() == BLACK_DISPLAY_KEY

    def recording_frame(self, global_frame: int, plane: int) -> np.ndarray:
        if self.recording_frame_starts is None or not self.recording_files:
            raise RuntimeError("No registered movie files are loaded.")

        total_frames = int(self.recording_frame_starts[-1])
        global_frame = max(0, min(int(global_frame), total_frames - 1))
        chunk_idx = int(np.searchsorted(self.recording_frame_starts, global_frame, side="right") - 1)
        local_frame = int(global_frame - self.recording_frame_starts[chunk_idx])

        if self.recording_chunk_cache is None or self.recording_chunk_cache[0] != chunk_idx:
            self.recording_chunk_cache = (
                chunk_idx,
                np.load(self.recording_files[chunk_idx], mmap_mode="r", allow_pickle=False),
            )
        chunk = self.recording_chunk_cache[1]
        return np.asarray(chunk[int(plane), local_frame])

    def update_array_selection(self) -> None:
        if self.info is None:
            return

        key = self.data_combo.currentText()
        if key == RECORDING_DISPLAY_KEY:
            if self.recording_shape is None or self.recording_frame_starts is None:
                self.status_label.setText("Registered movie: no recording files found")
                return
            nz, ny, nx = self.recording_shape
            current_plane = min(self.plane_slider.value(), nz - 1)
            self.plane_slider.blockSignals(True)
            self.plane_slider.setMaximum(nz - 1)
            self.plane_slider.setValue(current_plane)
            self.plane_slider.blockSignals(False)
            self.plane_slider.setEnabled(True)
            self.frame_slider.setEnabled(True)
            self.project_checkbox.setEnabled(False)
            self.projection_combo.setEnabled(False)
            total_frames = int(self.recording_frame_starts[-1])
            self.frame_label.setText(f"Frame: {self.frame_slider.value()} / {total_frames - 1}")
            self.current_array = None
            self.current_image_shape = (nz, ny, nx)
            self.status_label.setText(
                f"registered movie: shape z,t,y,x = {nz}, {total_frames}, {ny}, {nx}; "
                f"chunks = {len(self.recording_files)}"
            )
            self.update_play_button_states()
            return

        if key == BLACK_DISPLAY_KEY:
            shape = self.current_image_shape
            if shape is None:
                for display_key in DISPLAY_KEYS:
                    value = self.info.get(display_key)
                    if isinstance(value, np.ndarray) and value.ndim == 3:
                        shape = tuple(int(v) for v in value.shape)
                        break
            if shape is None:
                self.status_label.setText("Black: no image shape is available")
                return

            nz, ny, nx = shape
            self.current_array = np.zeros((nz, ny, nx), dtype=np.float32)
            self.current_image_shape = (int(nz), int(ny), int(nx))
            current_plane = min(self.plane_slider.value(), nz - 1)
            self.plane_slider.blockSignals(True)
            self.plane_slider.setMaximum(nz - 1)
            self.plane_slider.setValue(current_plane)
            self.plane_slider.blockSignals(False)
            self.plane_slider.setEnabled(not self.project_checkbox.isChecked())
            if self.recording_frame_starts is not None:
                total_frames = int(self.recording_frame_starts[-1])
                self.frame_slider.setEnabled(True)
                self.frame_label.setText(
                    f"Frame: {self.frame_slider.value()} / {total_frames - 1}"
                )
            else:
                self.frame_slider.setEnabled(False)
                self.frame_label.setText("Frame: -")
            self.project_checkbox.setEnabled(True)
            self.projection_combo.setEnabled(True)
            self.status_label.setText(f"{key}: shape z,y,x = {nz}, {ny}, {nx}")
            self.update_play_button_states()
            return

        arr = self.info[key]
        self.current_array = np.asarray(arr)

        nz, ny, nx = self.current_array.shape
        self.current_image_shape = (int(nz), int(ny), int(nx))
        current_plane = min(self.plane_slider.value(), nz - 1)
        self.plane_slider.blockSignals(True)
        self.plane_slider.setMaximum(nz - 1)
        self.plane_slider.setValue(current_plane)
        self.plane_slider.blockSignals(False)
        self.plane_slider.setEnabled(not self.project_checkbox.isChecked())
        if self.recording_frame_starts is not None:
            total_frames = int(self.recording_frame_starts[-1])
            self.frame_slider.setEnabled(True)
            self.frame_label.setText(
                f"Frame: {self.frame_slider.value()} / {total_frames - 1}"
            )
        else:
            self.frame_slider.setEnabled(False)
            self.frame_label.setText("Frame: -")
        self.project_checkbox.setEnabled(True)
        self.projection_combo.setEnabled(True)

        self.status_label.setText(f"{key}: shape z,y,x = {nz}, {ny}, {nx}; dtype = {self.current_array.dtype}")
        self.update_play_button_states()

    def on_display_changed(self) -> None:
        self.update_array_selection()
        self.update_projection_controls_visibility()
        self.update_image(preserve_view=True)

    def on_secondary_panel_mode_changed(self, _mode: str) -> None:
        if self.secondary_panel_mode() == "Accepted neuropil" and not self.masks_checkbox.isChecked():
            self.masks_checkbox.setChecked(True)
            return
        self.on_display_changed()

    def on_motion_correction_toggled(self) -> None:
        if self.motion_checkbox.isChecked():
            self.update_motion_correction_plot()
            return

        if self.selected_roi_idx is not None:
            self.update_trace_plot(self.selected_roi_idx)
        else:
            self.clear_trace_plot("F trace: select an ROI")

    def update_projection_controls_visibility(self) -> None:
        visible = self.project_checkbox.isChecked() and not self.is_recording_selected()
        self.projection_label.setVisible(visible)
        self.projection_combo.setVisible(visible)

    def update_play_button_states(self) -> None:
        plane_can_play = self.plane_slider.isEnabled() and self.plane_slider.maximum() > self.plane_slider.minimum()
        frame_can_play = (
            self.frame_slider.isEnabled()
            and self.recording_frame_starts is not None
            and self.frame_slider.maximum() > self.frame_slider.minimum()
        )

        if not plane_can_play:
            self.stop_plane_play()
        if not frame_can_play:
            self.stop_frame_play()

        self.plane_play_button.setEnabled(plane_can_play)
        self.frame_play_button.setEnabled(frame_can_play)

    def stop_plane_play(self) -> None:
        if self.plane_timer is not None and self.plane_timer.isActive():
            self.plane_timer.stop()
        if hasattr(self, "plane_play_button"):
            self.plane_play_button.setText("Play")

    def stop_frame_play(self) -> None:
        if self.frame_timer is not None and self.frame_timer.isActive():
            self.frame_timer.stop()
        if hasattr(self, "frame_play_button"):
            self.frame_play_button.setText("Play")

    def toggle_plane_play(self) -> None:
        if self.plane_timer is None:
            return
        if self.plane_timer.isActive():
            self.stop_plane_play()
            return
        if not self.plane_slider.isEnabled() or self.plane_slider.maximum() <= self.plane_slider.minimum():
            self.update_play_button_states()
            return
        self.plane_play_button.setText("Pause")
        self.plane_timer.start()

    def toggle_frame_play(self) -> None:
        if self.frame_timer is None:
            return
        if self.frame_timer.isActive():
            self.stop_frame_play()
            return
        if self.recording_frame_starts is None or self.frame_slider.maximum() <= self.frame_slider.minimum():
            self.update_play_button_states()
            return
        if not self.is_recording_selected() and RECORDING_DISPLAY_KEY in [
            self.data_combo.itemText(i) for i in range(self.data_combo.count())
        ]:
            self.data_combo.setCurrentText(RECORDING_DISPLAY_KEY)
        self.frame_play_button.setText("Pause")
        self.frame_timer.start()

    def advance_plane(self) -> None:
        if not self.plane_slider.isEnabled() or self.plane_slider.maximum() <= self.plane_slider.minimum():
            self.stop_plane_play()
            self.update_play_button_states()
            return
        next_value = self.plane_slider.value() + 1
        if next_value > self.plane_slider.maximum():
            next_value = self.plane_slider.minimum()
        self.plane_slider.setValue(next_value)

    def advance_frame(self) -> None:
        if self.recording_frame_starts is None or self.frame_slider.maximum() <= self.frame_slider.minimum():
            self.stop_frame_play()
            self.update_play_button_states()
            return
        if not self.is_recording_selected() and RECORDING_DISPLAY_KEY in [
            self.data_combo.itemText(i) for i in range(self.data_combo.count())
        ]:
            self.data_combo.setCurrentText(RECORDING_DISPLAY_KEY)
        next_value = self.frame_slider.value() + 1
        if next_value > self.frame_slider.maximum():
            next_value = self.frame_slider.minimum()
        self.frame_slider.setValue(next_value)

    def on_plane_slider_moved(self, value: int) -> None:
        self.plane_label.setText(f"Plane: {value}")

    def on_plane_changed(self) -> None:
        self.update_image(preserve_view=True)

    def on_frame_slider_moved(self, value: int) -> None:
        if self.recording_frame_starts is not None:
            total_frames = int(self.recording_frame_starts[-1])
            self.frame_label.setText(f"Frame: {value} / {total_frames - 1}")
        else:
            self.frame_label.setText(f"Frame: {value}")
        self.update_trace_cursor(value)

    def on_frame_changed(self) -> None:
        if self.recording_frame_starts is not None and not self.is_recording_selected():
            self.data_combo.setCurrentText(RECORDING_DISPLAY_KEY)
            self.update_trace_cursor(self.frame_slider.value())
            return
        self.update_image(preserve_view=True)
        self.update_trace_cursor(self.frame_slider.value())

    def is_left_mouse_button(self, button) -> bool:
        return button in (1, MouseButton.LEFT)

    def is_right_mouse_button(self, button) -> bool:
        return button in (3, MouseButton.RIGHT)

    def current_image(self) -> tuple[np.ndarray, str]:
        if self.is_recording_selected():
            if self.recording_shape is None or self.recording_frame_starts is None:
                raise RuntimeError("No registered movie loaded.")
            plane = self.plane_slider.value()
            frame = self.frame_slider.value()
            image = self.recording_frame(frame, plane)
            title = f"registered movie, plane {plane}, frame {frame}"
            return image, title

        if self.current_array is None:
            raise RuntimeError("No image loaded.")

        key = self.data_combo.currentText()
        if self.project_checkbox.isChecked():
            mode = self.projection_combo.currentText()
            if mode == "Max":
                image = np.nanmax(self.current_array, axis=0)
            else:
                image = np.nanmean(self.current_array, axis=0)
            title = f"{key} {mode.lower()} projection across planes"
        else:
            plane = self.plane_slider.value()
            image = self.current_array[plane]
            title = f"{key}, plane {plane}"

        return image, title

    def roi_matches_panel(self, roi_idx: int, accepted_panel: bool) -> bool:
        if self.iscell is None or roi_idx >= len(self.iscell):
            return accepted_panel
        is_accepted = bool(self.iscell[roi_idx, 0] > 0)
        return is_accepted == accepted_panel

    def secondary_panel_mode(self) -> str:
        if not hasattr(self, "secondary_panel_combo"):
            return "Non-accepted cells"
        return self.secondary_panel_combo.currentText()

    def roi_mask_coords(self, stat: dict, mask_kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        coord_key = "npcoords" if mask_kind == "neuropil" else "coords"
        coords = stat.get(coord_key)
        if coords is None or len(coords) != 3:
            return None

        z, y, x = [np.asarray(c) for c in coords]
        if z.size == 0 or y.size != z.size or x.size != z.size:
            return None
        return z, y, x

    def has_neuropil_coords(self) -> bool:
        if self.stats is None:
            return False
        for stat in self.stats:
            coords = self.roi_mask_coords(stat, "neuropil")
            if coords is not None:
                return True
        return False

    def build_mask_overlay(
        self,
        image_shape: tuple[int, int],
        accepted_panel: bool,
        mask_kind: str = "roi",
    ) -> tuple[np.ndarray | None, np.ndarray]:
        overlay, id_map = self.base_mask_overlay(image_shape, accepted_panel, mask_kind=mask_kind)
        if (
            self.selected_roi_idx is None
            or self.stats is None
            or not (0 <= self.selected_roi_idx < len(self.stats))
            or not self.roi_matches_panel(self.selected_roi_idx, accepted_panel)
            or not self.masks_checkbox.isChecked()
        ):
            return overlay, id_map

        selected_overlay = None if overlay is None else overlay.copy()
        selected_id_map = id_map.copy()
        selected_overlay, selected_id_map = self.add_selected_roi_to_overlay(
            selected_overlay,
            selected_id_map,
            image_shape,
            accepted_panel,
            mask_kind=mask_kind,
        )
        return selected_overlay, selected_id_map

    def base_mask_overlay(
        self,
        image_shape: tuple[int, int],
        accepted_panel: bool,
        mask_kind: str = "roi",
    ) -> tuple[np.ndarray | None, np.ndarray]:
        if (
            self.stats is None
            or self.roi_colors is None
            or not self.masks_checkbox.isChecked()
        ):
            return None, np.full(image_shape, -1, dtype=np.int32)

        projected = self.project_checkbox.isChecked() and not self.is_recording_selected()
        plane = None if projected else int(self.plane_slider.value())
        color_mode = self.roi_color_overlay_mode if self.correlation_colors is not None else "Random"
        cache_key = (
            tuple(image_shape),
            bool(accepted_panel),
            mask_kind,
            plane,
            color_mode,
            self.correlation_seed_roi_idx,
        )
        cached = self.overlay_cache.get(cache_key)
        if cached is not None:
            return cached

        ny, nx = image_shape
        overlay = np.zeros((ny, nx, 4), dtype=np.float32)
        id_map = np.full((ny, nx), -1, dtype=np.int32)
        alpha = 0.34 if mask_kind == "neuropil" else 0.55

        for roi_idx, stat in enumerate(self.stats):
            if not self.roi_matches_panel(roi_idx, accepted_panel):
                continue
            coords = self.roi_mask_coords(stat, mask_kind)
            if coords is None:
                continue

            z, y, x = coords
            if projected:
                keep = np.ones(z.shape, dtype=bool)
            else:
                keep = z == plane
            if not np.any(keep):
                continue

            yy = y[keep].astype(np.int64, copy=False)
            xx = x[keep].astype(np.int64, copy=False)
            in_bounds = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
            if not np.any(in_bounds):
                continue

            yy = yy[in_bounds]
            xx = xx[in_bounds]
            overlay[yy, xx, :3] = self.roi_mask_color(roi_idx)
            overlay[yy, xx, 3] = alpha
            id_map[yy, xx] = roi_idx

        if np.any(id_map >= 0):
            result = (overlay, id_map)
        else:
            result = (None, id_map)
        self.overlay_cache[cache_key] = result
        return result

    def roi_mask_color(self, roi_idx: int) -> np.ndarray:
        if self.correlation_colors is not None and 0 <= roi_idx < self.correlation_colors.shape[0]:
            return self.correlation_colors[roi_idx]
        if self.roi_colors is not None and 0 <= roi_idx < self.roi_colors.shape[0]:
            return self.roi_colors[roi_idx]
        return np.array([0.8, 0.8, 0.8], dtype=np.float32)

    def add_selected_roi_to_overlay(
        self,
        overlay: np.ndarray | None,
        id_map: np.ndarray,
        image_shape: tuple[int, int],
        accepted_panel: bool,
        mask_kind: str = "roi",
    ) -> tuple[np.ndarray | None, np.ndarray]:
        if self.stats is None or self.selected_roi_idx is None:
            return overlay, id_map
        if not self.roi_matches_panel(self.selected_roi_idx, accepted_panel):
            return overlay, id_map

        stat = self.stats[self.selected_roi_idx]
        coords = self.roi_mask_coords(stat, mask_kind)
        if coords is None:
            return overlay, id_map

        ny, nx = image_shape
        z, y, x = coords
        projected = self.project_checkbox.isChecked() and not self.is_recording_selected()
        if projected:
            keep = np.ones(z.shape, dtype=bool)
        else:
            keep = z == self.plane_slider.value()
        if not np.any(keep):
            return overlay, id_map

        yy = y[keep].astype(np.int64, copy=False)
        xx = x[keep].astype(np.int64, copy=False)
        in_bounds = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
        yy = yy[in_bounds]
        xx = xx[in_bounds]
        if not yy.size:
            return overlay, id_map

        if overlay is None:
            overlay = np.zeros((ny, nx, 4), dtype=np.float32)
        overlay[yy, xx, :3] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        overlay[yy, xx, 3] = 0.95
        id_map[yy, xx] = self.selected_roi_idx
        return overlay, id_map

    def update_image(self, preserve_view: bool = False) -> None:
        if self.current_array is None and not self.is_recording_selected():
            return

        old_xlim = None
        old_ylim = None
        if preserve_view and self.image_axes:
            old_xlim = self.image_axes[0].get_xlim()
            old_ylim = self.image_axes[0].get_ylim()

        image, title = self.current_image()
        image = np.asarray(image)

        if self.is_black_background_selected():
            lo, hi = 0.0, 1.0
        else:
            lo, hi = np.nanpercentile(image, [1, 99.8])
            if hi <= lo:
                lo = float(np.nanmin(image))
                hi = float(np.nanmax(image))

        self.plane_label.setText(f"Plane: {self.plane_slider.value()}")
        if self.is_recording_selected():
            self.plane_slider.setEnabled(True)
            if self.recording_frame_starts is not None:
                total_frames = int(self.recording_frame_starts[-1])
                self.frame_label.setText(f"Frame: {self.frame_slider.value()} / {total_frames - 1}")
        else:
            self.plane_slider.setEnabled(not self.project_checkbox.isChecked())
        self.update_play_button_states()

        self.figure.clear()
        self.figure.patch.set_facecolor(GUI_BG)
        self.current_id_maps = {}
        self.image_axes = []
        self.image_axes_panels = {}
        self.image_axes_roles = {}

        show_nonaccepted = not hasattr(self, "show_nonaccepted_checkbox") or self.show_nonaccepted_checkbox.isChecked()
        secondary_mode = self.secondary_panel_mode()
        ax_left = self.figure.add_subplot(1, 2, 1) if show_nonaccepted else self.figure.add_subplot(1, 1, 1)
        self.current_axes = ax_left
        self.image_axes = [ax_left]
        self.image_axes_panels = {ax_left: True}
        self.image_axes_roles = {ax_left: "accepted"}

        panel_specs = [(ax_left, True, "Accepted cells", "roi", "accepted")]
        if show_nonaccepted:
            ax_right = self.figure.add_subplot(1, 2, 2, sharex=ax_left, sharey=ax_left)
            self.image_axes.append(ax_right)
            if secondary_mode == "Accepted neuropil":
                self.image_axes_panels[ax_right] = True
                self.image_axes_roles[ax_right] = "neuropil"
                panel_specs.append((ax_right, True, "Accepted neuropil", "neuropil", "neuropil"))
            else:
                self.image_axes_panels[ax_right] = False
                self.image_axes_roles[ax_right] = "nonaccepted"
                panel_specs.append((ax_right, False, "Non-accepted cells", "roi", "nonaccepted"))

        for ax, accepted_panel, panel_name, mask_kind, role in panel_specs:
            ax.set_facecolor(PLOT_BG)
            ax.imshow(image, cmap="gray", vmin=lo, vmax=hi, aspect="equal")
            overlay, id_map = self.build_mask_overlay(image.shape, accepted_panel, mask_kind=mask_kind)
            self.current_id_maps[ax] = id_map
            if overlay is not None:
                ax.imshow(overlay, interpolation="nearest", aspect="equal")
            elif role == "neuropil":
                message = "No saved npcoords for accepted ROIs" if not self.has_neuropil_coords() else "No neuropil pixels on this plane"
                ax.text(
                    0.5,
                    0.5,
                    message,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color=TEXT_FG,
                    fontsize=10,
                    bbox={"facecolor": PLOT_BG, "edgecolor": GRID_FG, "alpha": 0.82, "pad": 6},
                )
            title = panel_name
            overlay_label = self.roi_color_overlay_title()
            if overlay_label is not None:
                title = f"{panel_name}; {overlay_label}"
            ax.set_title(title, color=TEXT_FG)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.xaxis.label.set_color(TEXT_FG)
            ax.yaxis.label.set_color(TEXT_FG)
            ax.tick_params(colors=TEXT_FG)
            for spine in ax.spines.values():
                spine.set_color(GRID_FG)

        if old_xlim is not None and old_ylim is not None:
            for ax in self.image_axes:
                ax.set_xlim(old_xlim)
                ax.set_ylim(old_ylim)
        self.canvas.draw_idle()

    def roi_color_overlay_title(self) -> str | None:
        if self.correlation_colors is None:
            return None
        if self.roi_color_overlay_mode == "Correlation" and self.correlation_seed_roi_idx is not None:
            return f"corr with ROI {self.correlation_seed_roi_idx}"
        if self.roi_colorbar_label:
            return self.roi_colorbar_label
        return None

    def clear_correlation_colorbar(self) -> None:
        if not hasattr(self, "correlation_colorbar_figure"):
            return
        self.correlation_colorbar_canvas.setVisible(False)
        self.correlation_colorbar_figure.clear()
        self.correlation_colorbar_figure.patch.set_facecolor(PANEL_BG)
        ax = self.correlation_colorbar_figure.add_axes([0.08, 0.36, 0.84, 0.30])
        ax.set_facecolor(PANEL_BG)
        ax.set_axis_off()
        self.correlation_colorbar_canvas.draw_idle()

    def update_correlation_colorbar(self) -> None:
        if (
            self.correlation_colors is None
            or self.correlation_values is None
            or self.correlation_color_limits is None
            or not hasattr(self, "correlation_colorbar_figure")
        ):
            self.clear_correlation_colorbar()
            return

        try:
            from matplotlib import cm
            from matplotlib.colors import Normalize, TwoSlopeNorm

            self.correlation_colorbar_canvas.setVisible(True)
            self.correlation_colorbar_figure.clear()
            self.correlation_colorbar_figure.patch.set_facecolor(PANEL_BG)
            ax = self.correlation_colorbar_figure.add_axes([0.10, 0.40, 0.80, 0.28])
            ax.set_facecolor(PANEL_BG)

            vmin, vmax = self.correlation_color_limits
            if self.roi_colorbar_center_zero and vmin < 0.0 < vmax:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)
            mappable = cm.ScalarMappable(norm=norm, cmap=self.roi_colorbar_cmap_name)
            mappable.set_array([])
            cbar = self.correlation_colorbar_figure.colorbar(
                mappable,
                cax=ax,
                orientation="horizontal",
            )
            cbar.set_label(self.roi_colorbar_label or "ROI color value", labelpad=1, fontsize=9)
            cbar.ax.xaxis.label.set_color(TEXT_FG)
            ticks = [vmin, vmax] if not (vmin < 0.0 < vmax) else [vmin, 0.0, vmax]
            cbar.set_ticks(ticks)
            cbar.ax.set_xticklabels([f"{tick:.2f}" for tick in ticks])
            cbar.ax.tick_params(colors=TEXT_FG)
            cbar.ax.tick_params(labelsize=9, pad=1)
            cbar.outline.set_edgecolor(GRID_FG)
            self.correlation_colorbar_canvas.draw_idle()
        except Exception:
            self.clear_correlation_colorbar()
            return

    def clear_trace_plot(self, message: str) -> None:
        if not hasattr(self, "trace_figure"):
            return
        if hasattr(self, "trace_group") and not self.motion_checkbox.isChecked():
            self.trace_group.setTitle("F trace")
        self.trace_cursor_lines = []
        self.trace_zoom_axes = None
        self.trace_figure.clear()
        self.trace_figure.patch.set_facecolor(GUI_BG)
        ax = self.trace_figure.add_subplot(111)
        ax.set_facecolor(PLOT_BG)
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color=TEXT_FG)
        ax.set_axis_off()
        self.trace_canvas.draw_idle()

    def style_trace_axis(self, ax, xlabel: str | None = None, ylabel: str | None = None) -> None:
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=8, labelpad=1)
            ax.xaxis.label.set_color(TEXT_FG)
        else:
            ax.set_xlabel("")
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8, labelpad=2)
            ax.yaxis.label.set_color(TEXT_FG)
        else:
            ax.set_ylabel("")
        ax.tick_params(colors=TEXT_FG, labelsize=8, pad=1, length=3)
        for spine in ax.spines.values():
            spine.set_color(GRID_FG)
        ax.grid(True, alpha=0.2, color=GRID_FG)

    def on_trace_scroll_zoom(self, event) -> None:
        if event.inaxes is None or event.inaxes not in self.trace_figure.axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.set_trace_full_view_checked(False)

        scale = 0.82 if event.button == "up" else 1.22
        ax = event.inaxes
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        x_span = max(abs(x_right - x_left), 1e-9)
        y_span = max(abs(y_high - y_low), 1e-9)

        new_x_span = max(x_span * scale, 5.0)
        x_fraction = float(np.clip((event.xdata - x_left) / x_span, 0.0, 1.0))
        new_x_left = event.xdata - new_x_span * x_fraction
        new_x_right = event.xdata + new_x_span * (1.0 - x_fraction)
        new_xlim = self.clamp_trace_xlim((new_x_left, new_x_right))

        self.set_trace_xlim_for_shared_axes(ax, new_xlim)

        new_y_span = y_span * scale
        y_fraction = float(np.clip((event.ydata - y_low) / y_span, 0.0, 1.0))
        ax.set_ylim(event.ydata - new_y_span * y_fraction, event.ydata + new_y_span * (1.0 - y_fraction))
        self.trace_canvas.draw_idle()

    def set_trace_full_view_checked(self, checked: bool) -> None:
        if not hasattr(self, "trace_full_view_checkbox"):
            return
        self.trace_full_view_checkbox.blockSignals(True)
        self.trace_full_view_checkbox.setChecked(checked)
        self.trace_full_view_checkbox.blockSignals(False)

    def on_trace_full_view_toggled(self, state: int) -> None:
        if state == Qt.Checked:
            self.reset_trace_view()

    def selected_trace_labels(self) -> set[str]:
        if not hasattr(self, "trace_visibility_checkboxes"):
            return set()
        return {
            label
            for label, checkbox in self.trace_visibility_checkboxes.items()
            if checkbox.isChecked()
        }

    def on_trace_visibility_changed(self, _state: int) -> None:
        if self.motion_checkbox.isChecked():
            return
        if self.selected_roi_idx is not None:
            self.set_trace_full_view_checked(True)
            self.update_trace_plot(self.selected_roi_idx)

    def trace_data_xlim(self) -> tuple[float, float] | None:
        if not hasattr(self, "trace_figure"):
            return None
        x_parts = []
        for ax in self.trace_figure.axes:
            for line in ax.lines:
                if line.get_label() == "_nolegend_":
                    continue
                xdata = np.asarray(line.get_xdata(), dtype=float)
                xdata = xdata[np.isfinite(xdata)]
                if xdata.size:
                    x_parts.append(xdata)
        if not x_parts:
            return None
        x = np.concatenate(x_parts)
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if x_max <= x_min:
            x_max = x_min + 1.0
        return x_min, x_max

    def clamp_trace_xlim(self, xlim: tuple[float, float]) -> tuple[float, float]:
        full_xlim = self.trace_data_xlim()
        if full_xlim is None:
            return xlim
        full_left, full_right = full_xlim
        left, right = xlim
        span = max(float(right - left), 1e-9)
        full_span = max(float(full_right - full_left), 1e-9)
        if span >= full_span:
            return full_left, full_right
        left = min(max(float(left), full_left), full_right - span)
        return left, left + span

    def set_trace_xlim_for_shared_axes(self, source_ax, xlim: tuple[float, float]) -> None:
        shared_x_axes = source_ax.get_shared_x_axes()
        for ax in self.trace_figure.axes:
            if ax is source_ax or shared_x_axes.joined(source_ax, ax):
                ax.set_xlim(*xlim)

    def reset_trace_view(self) -> None:
        if not hasattr(self, "trace_figure"):
            return
        any_data = False
        for ax in self.trace_figure.axes:
            data_lines = [line for line in ax.lines if line.get_label() != "_nolegend_"]
            if not data_lines:
                continue
            x_parts = []
            y_parts = []
            for line in data_lines:
                xdata = np.asarray(line.get_xdata(), dtype=float)
                ydata = np.asarray(line.get_ydata(), dtype=float)
                mask = np.isfinite(xdata) & np.isfinite(ydata)
                if np.any(mask):
                    x_parts.append(xdata[mask])
                    y_parts.append(ydata[mask])
            if not x_parts:
                continue
            x = np.concatenate(x_parts)
            y = np.concatenate(y_parts)
            x_min = float(np.min(x))
            x_max = float(np.max(x))
            y_min = float(np.min(y))
            y_max = float(np.max(y))
            if x_max <= x_min:
                x_max = x_min + 1.0
            if y_max <= y_min:
                margin = max(abs(y_min) * 0.05, 1.0)
            else:
                margin = max((y_max - y_min) * 0.05, 1.0)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min - margin, y_max + margin)
            any_data = True
        if any_data:
            self.set_trace_full_view_checked(True)
            self.trace_canvas.draw_idle()

    def on_trace_press(self, event) -> None:
        if event.inaxes is None or event.inaxes not in self.trace_figure.axes:
            return
        if event.x is None or event.y is None or event.xdata is None:
            return
        if not self.is_left_mouse_button(event.button):
            return
        self.trace_pan_start = {
            "axes": event.inaxes,
            "xpix": event.x,
            "xlim": event.inaxes.get_xlim(),
            "moved": False,
        }

    def on_trace_motion(self, event) -> None:
        if self.trace_pan_start is None or event.x is None:
            return
        ax = self.trace_pan_start["axes"]
        dx_pixels = event.x - int(self.trace_pan_start["xpix"])
        if abs(dx_pixels) < self.drag_pixel_threshold:
            return
        self.set_trace_full_view_checked(False)
        self.trace_pan_start["moved"] = True
        start_xlim = self.trace_pan_start["xlim"]
        x_per_pixel = (start_xlim[1] - start_xlim[0]) / max(ax.bbox.width, 1.0)
        dx_data = dx_pixels * x_per_pixel
        new_xlim = self.clamp_trace_xlim((start_xlim[0] - dx_data, start_xlim[1] - dx_data))
        self.set_trace_xlim_for_shared_axes(ax, new_xlim)
        self.trace_canvas.draw_idle()

    def on_trace_release(self, _event) -> None:
        self.trace_pan_start = None

    def update_motion_correction_plot(self) -> None:
        if not hasattr(self, "trace_figure"):
            return
        if self.motion_shifts is None or self.motion_shifts.size == 0:
            self.clear_trace_plot("Motion correction: no offsets*.npy files found")
            return

        shifts = np.asarray(self.motion_shifts, dtype=float)
        frames = np.arange(shifts.shape[0])
        y_shift = shifts[:, 1]
        x_shift = shifts[:, 2]
        xy_motion = np.sqrt(y_shift**2 + x_shift**2)
        frame = min(max(self.frame_slider.value(), 0), shifts.shape[0] - 1)

        self.trace_group.setTitle("Motion correction")
        self.trace_figure.clear()
        self.trace_figure.patch.set_facecolor(GUI_BG)
        self.trace_cursor_lines = []
        axes = self.trace_figure.subplots(3, 1, sharex=True)
        self.trace_zoom_axes = axes[-1]
        plot_specs = [
            (axes[0], y_shift, "#61a5ff", "Y shift", "Y pixels"),
            (axes[1], x_shift, "#ffb74d", "X shift", "X pixels"),
            (axes[2], xy_motion, "white", "X-Y motion", "pixels"),
        ]

        for ax, values, color, label, ylabel in plot_specs:
            ax.set_facecolor(PLOT_BG)
            ax.plot(frames, values, color=color, linewidth=0.8, label=label)
            cursor = ax.axvline(
                frame,
                color="#ffd54f",
                linewidth=1.4,
                linestyle="--",
                label="_nolegend_",
            )
            self.trace_cursor_lines.append(cursor)
            self.style_trace_axis(ax, ylabel=ylabel)
            legend = ax.legend(loc="upper right", facecolor=PLOT_BG, edgecolor=GRID_FG, fontsize=8)
            for text in legend.get_texts():
                text.set_color(TEXT_FG)

        axes[0].set_title(
            f"Motion correction shifts; frame {frame}; "
            f"y={y_shift[frame]:.3g}, x={x_shift[frame]:.3g}, x-y={xy_motion[frame]:.3g} pixels",
            color=TEXT_FG,
            fontsize=10,
            pad=2,
        )
        axes[-1].set_xlabel("Frame", fontsize=8, labelpad=1)
        axes[-1].xaxis.label.set_color(TEXT_FG)
        self.trace_figure.subplots_adjust(left=0.045, right=0.995, top=0.91, bottom=0.12, hspace=0.14)
        self.trace_canvas.draw_idle()

    def trace_for_roi(self, trace_key: str, roi_idx: int) -> np.ndarray | None:
        trace_array = self.traces.get(trace_key)
        roi_axis = self.trace_roi_axes.get(trace_key)
        if trace_array is None or roi_axis is None:
            return None
        if self.stats is None or not (0 <= roi_idx < len(self.stats)):
            return None
        if roi_axis == 0:
            trace = trace_array[roi_idx, :]
        else:
            trace = trace_array[:, roi_idx]
        return np.asarray(trace).squeeze()

    def update_trace_plot(self, roi_idx: int) -> None:
        if self.motion_checkbox.isChecked():
            self.update_motion_correction_plot()
            return

        f_trace = self.trace_for_roi("F", roi_idx)
        fneu_trace = self.trace_for_roi("Fneu", roi_idx)
        spks_trace = self.trace_for_roi("spks", roi_idx)
        available = {
            "F": f_trace,
            "Neuropil": fneu_trace,
            "Deconvolved": spks_trace,
        }
        available = {key: value for key, value in available.items() if value is not None}
        if not available:
            self.clear_trace_plot("Trace: no F.npy/Fneu.npy/spks.npy files found for this Suite3D output")
            return
        selected_labels = self.selected_trace_labels()
        visible = {key: value for key, value in available.items() if key in selected_labels}
        if not visible:
            self.clear_trace_plot(f"Trace: select F, Neuropil, or Deconvolved for ROI {roi_idx}")
            return
        bad = [key for key, value in visible.items() if value.ndim != 1 or value.size == 0]
        if bad:
            self.clear_trace_plot(f"Trace: ROI {roi_idx} has invalid trace(s): {', '.join(bad)}")
            return

        self.trace_figure.clear()
        self.trace_figure.patch.set_facecolor(GUI_BG)
        self.trace_cursor_lines = []
        self.trace_zoom_axes = None
        self.trace_group.setTitle("F trace")
        ax = self.trace_figure.add_subplot(111)
        self.trace_zoom_axes = ax
        ax.set_facecolor(PLOT_BG)

        trace_colors = {
            "F": "lime",
            "Neuropil": "red",
            "Deconvolved": "white",
        }
        for label, trace in visible.items():
            frames = np.arange(trace.size)
            ax.plot(frames, trace, color=trace_colors[label], linewidth=0.8, label=label)

        cursor = ax.axvline(
            self.frame_slider.value(),
            color="#ffd54f",
            linewidth=1.5,
            linestyle="--",
            label="_nolegend_",
        )
        self.trace_cursor_lines = [cursor]

        ax.set_title(f"ROI {roi_idx} traces", color=TEXT_FG, fontsize=10, pad=2)
        self.style_trace_axis(ax, xlabel="Frame", ylabel="Signal")
        legend = ax.legend(loc="upper right", facecolor=PLOT_BG, edgecolor=GRID_FG, fontsize=8)
        for text in legend.get_texts():
            text.set_color(TEXT_FG)
        self.trace_figure.subplots_adjust(left=0.045, right=0.995, top=0.90, bottom=0.14)
        self.trace_canvas.draw_idle()

    def update_trace_cursor(self, frame: int) -> None:
        if not self.trace_cursor_lines:
            return
        for line in self.trace_cursor_lines:
            line.set_xdata([frame, frame])
        if self.motion_checkbox.isChecked() and self.motion_shifts is not None and self.trace_figure.axes:
            shifts = np.asarray(self.motion_shifts, dtype=float)
            frame = min(max(int(frame), 0), shifts.shape[0] - 1)
            y_shift = shifts[frame, 1]
            x_shift = shifts[frame, 2]
            xy_motion = float(np.sqrt(y_shift**2 + x_shift**2))
            self.trace_figure.axes[0].set_title(
                f"Motion correction shifts; frame {frame}; "
                f"y={y_shift:.3g}, x={x_shift:.3g}, x-y={xy_motion:.3g} pixels",
                color=TEXT_FG,
            )
        self.trace_canvas.draw_idle()

    def on_canvas_press(self, event) -> None:
        if event.inaxes is None or event.inaxes not in self.image_axes:
            return
        if event.xdata is None or event.ydata is None:
            return

        if self.is_left_mouse_button(event.button):
            mode = "pan"
        elif self.is_right_mouse_button(event.button):
            if self.image_axes_roles.get(event.inaxes) == "neuropil":
                self.status_label.setText("Neuropil panel: click to select the owning ROI; curate from the ROI panels.")
                return
            mode = "curate"
            self.clear_selection_rectangle()
        else:
            return

        self.drag_start = {
            "mode": mode,
            "axes": event.inaxes,
            "xdata": float(event.xdata),
            "ydata": float(event.ydata),
            "xpix": event.x,
            "ypix": event.y,
            "xlim": event.inaxes.get_xlim(),
            "ylim": event.inaxes.get_ylim(),
            "accepted_panel": self.image_axes_panels.get(event.inaxes),
            "moved": False,
        }

    def on_canvas_motion(self, event) -> None:
        if self.drag_start is None:
            return

        if event.x is None or event.y is None:
            return

        dx_pixels = abs(event.x - int(self.drag_start["xpix"]))
        dy_pixels = abs(event.y - int(self.drag_start["ypix"]))
        if dx_pixels < self.drag_pixel_threshold and dy_pixels < self.drag_pixel_threshold:
            return

        self.drag_start["moved"] = True
        mode = self.drag_start.get("mode")
        if mode == "pan":
            axes = self.drag_start["axes"]
            signed_dx_pixels = event.x - int(self.drag_start["xpix"])
            signed_dy_pixels = event.y - int(self.drag_start["ypix"])
            start_xlim = self.drag_start["xlim"]
            start_ylim = self.drag_start["ylim"]
            x_per_pixel = (start_xlim[1] - start_xlim[0]) / axes.bbox.width
            y_per_pixel = (start_ylim[1] - start_ylim[0]) / axes.bbox.height
            dx_data = signed_dx_pixels * x_per_pixel
            dy_data = signed_dy_pixels * y_per_pixel
            new_xlim = (start_xlim[0] - dx_data, start_xlim[1] - dx_data)
            new_ylim = (start_ylim[0] - dy_data, start_ylim[1] - dy_data)
            for ax in self.image_axes:
                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)
        elif mode == "curate":
            xdata, ydata = self.data_point_from_mouse_event(event, self.drag_start["axes"])
            self.update_selection_rectangle(float(xdata), float(ydata))
        self.canvas.draw_idle()

    def on_canvas_release(self, event) -> None:
        if self.drag_start is None:
            return

        drag_start = self.drag_start
        self.drag_start = None
        mode = drag_start.get("mode")
        if drag_start["moved"]:
            if mode != "curate":
                return
            if event.x is not None and event.y is not None:
                xdata, ydata = self.data_point_from_mouse_event(event, drag_start["axes"])
            else:
                xdata, ydata = float(drag_start["xdata"]), float(drag_start["ydata"])
            self.clear_selection_rectangle()
            self.move_rois_in_rectangle(
                drag_start["axes"],
                float(drag_start["xdata"]),
                float(drag_start["ydata"]),
                float(xdata),
                float(ydata),
            )
            return

        if mode == "curate":
            self.toggle_roi_at(drag_start["axes"], float(drag_start["xdata"]), float(drag_start["ydata"]))
        else:
            self.select_roi_at(
                drag_start["axes"],
                float(drag_start["xdata"]),
                float(drag_start["ydata"]),
            )

    def data_point_from_mouse_event(self, event, axes) -> tuple[float, float]:
        if event.xdata is not None and event.ydata is not None and event.inaxes is axes:
            return float(event.xdata), float(event.ydata)
        xdata, ydata = axes.transData.inverted().transform((event.x, event.y))
        return float(xdata), float(ydata)

    def update_selection_rectangle(self, xdata: float, ydata: float) -> None:
        if self.drag_start is None:
            return
        axes = self.drag_start["axes"]
        x0 = float(self.drag_start["xdata"])
        y0 = float(self.drag_start["ydata"])
        x = min(x0, xdata)
        y = min(y0, ydata)
        width = abs(xdata - x0)
        height = abs(ydata - y0)
        if self.selection_rect is None:
            self.selection_rect = Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor="#ffd54f",
                linewidth=1.5,
                linestyle="--",
                zorder=20,
            )
            axes.add_patch(self.selection_rect)
        else:
            self.selection_rect.set_xy((x, y))
            self.selection_rect.set_width(width)
            self.selection_rect.set_height(height)

    def clear_selection_rectangle(self) -> None:
        if self.selection_rect is None:
            return
        try:
            self.selection_rect.remove()
        except Exception:
            pass
        self.selection_rect = None
        if hasattr(self, "canvas"):
            self.canvas.draw_idle()

    def push_curation_undo_state(self) -> None:
        if self.iscell is None:
            return
        self.curation_undo_stack.append(
            {
                "iscell": np.asarray(self.iscell[:, 0]).copy(),
                "manual": dict(self.manual_curation_overrides),
                "neuropil_rejected": set(self.neuropil_rejected_roi_indices),
            }
        )
        if len(self.curation_undo_stack) > 50:
            self.curation_undo_stack.pop(0)

    def undo_curation_action(self) -> None:
        if self.iscell is None or not self.curation_undo_stack:
            return
        previous = self.curation_undo_stack.pop()
        previous_iscell = previous["iscell"]
        if previous_iscell.shape[0] != self.iscell.shape[0]:
            return
        self.iscell[:, 0] = previous_iscell.astype(self.iscell.dtype, copy=False)
        self.manual_curation_overrides = dict(previous["manual"])
        self.neuropil_rejected_roi_indices = set(previous.get("neuropil_rejected", set()))
        self.overlay_cache = {}
        self.update_neuropil_rejection_undo_button()
        self.update_curation_histograms()
        self.update_image(preserve_view=True)

    def apply_roi_acceptance(self, roi_indices: np.ndarray, accepted: bool) -> None:
        if self.iscell is None or roi_indices.size == 0:
            return
        roi_indices = np.unique(roi_indices.astype(np.int64, copy=False))
        roi_indices = roi_indices[(roi_indices >= 0) & (roi_indices < self.iscell.shape[0])]
        if roi_indices.size == 0:
            return
        if accepted and self.neuropil_rejected_roi_indices:
            blocked = np.array(
                [int(roi_idx) in self.neuropil_rejected_roi_indices for roi_idx in roi_indices],
                dtype=bool,
            )
            if np.any(blocked):
                roi_indices = roi_indices[~blocked]
                self.status_label.setText("Use Undo next to Reject Fneu > F to accept those ROIs again.")
                if roi_indices.size == 0:
                    return
        new_value = 1 if accepted else 0
        already_visible = np.all(self.iscell[roi_indices, 0] == new_value)
        already_manual = all(self.manual_curation_overrides.get(int(roi_idx)) == bool(accepted) for roi_idx in roi_indices)
        if already_visible and already_manual:
            return
        self.push_curation_undo_state()
        for roi_idx in roi_indices:
            self.manual_curation_overrides[int(roi_idx)] = bool(accepted)
        self.iscell[roi_indices, 0] = new_value
        self.overlay_cache = {}
        self.update_curation_histograms()
        self.update_image(preserve_view=True)

    def toggle_roi_at(self, axes, xdata: float, ydata: float) -> None:
        roi_idx = self.roi_at(axes, xdata, ydata)
        if roi_idx is None or self.iscell is None:
            return
        accepted = not bool(self.iscell[roi_idx, 0] > 0)
        self.selected_roi_idx = roi_idx
        self.selected_panel_accepted = bool(accepted)
        self.selected_mask_kind = "roi"
        self.update_show_3d_button_text()
        self.apply_roi_acceptance(np.array([roi_idx], dtype=np.int64), accepted)
        self.update_trace_plot(roi_idx)

    def move_rois_in_rectangle(self, axes, x0: float, y0: float, x1: float, y1: float) -> None:
        accepted_panel = self.image_axes_panels.get(axes)
        if accepted_panel is None:
            return
        roi_indices = self.roi_indices_in_rectangle(axes, x0, y0, x1, y1)
        if roi_indices.size == 0:
            return
        self.apply_roi_acceptance(roi_indices, accepted=not bool(accepted_panel))

    def roi_indices_in_rectangle(self, axes, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        id_map = self.current_id_maps.get(axes)
        if id_map is None:
            return np.array([], dtype=np.int64)
        ny, nx = id_map.shape
        xmin = max(0, int(np.floor(min(x0, x1))))
        xmax = min(nx - 1, int(np.ceil(max(x0, x1))))
        ymin = max(0, int(np.floor(min(y0, y1))))
        ymax = min(ny - 1, int(np.ceil(max(y0, y1))))
        if xmin > xmax or ymin > ymax:
            return np.array([], dtype=np.int64)
        roi_indices = np.unique(id_map[ymin : ymax + 1, xmin : xmax + 1])
        return roi_indices[roi_indices >= 0].astype(np.int64, copy=False)

    def roi_at(self, axes, xdata: float, ydata: float) -> int | None:
        if axes not in self.current_id_maps:
            return None

        id_map = self.current_id_maps[axes]
        x = int(round(xdata))
        y = int(round(ydata))
        ny, nx = id_map.shape
        if x < 0 or x >= nx or y < 0 or y >= ny:
            return None

        roi_idx = int(id_map[y, x])
        if roi_idx < 0:
            return None
        return roi_idx

    def select_roi_at(self, axes, xdata: float, ydata: float) -> None:
        roi_idx = self.roi_at(axes, xdata, ydata)
        if roi_idx is None:
            self.selected_roi_idx = None
            self.selected_mask_kind = "roi"
            self.update_show_3d_button_text()
            if not self.motion_checkbox.isChecked():
                self.clear_trace_plot("F trace: no ROI selected")
            self.update_image(preserve_view=True)
            return

        role = self.image_axes_roles.get(axes)
        mask_kind = "neuropil" if role == "neuropil" else "roi"
        self.select_roi_by_index(roi_idx, self.image_axes_panels.get(axes), mask_kind=mask_kind)

    def on_scroll_zoom(self, event) -> None:
        if event.inaxes is None or event.inaxes not in self.image_axes:
            return
        if event.xdata is None or event.ydata is None:
            return

        scale = 0.8 if event.button == "up" else 1.25
        xlim = event.inaxes.get_xlim()
        ylim = event.inaxes.get_ylim()
        x_mouse = float(event.xdata)
        y_mouse = float(event.ydata)

        new_xlim = (
            x_mouse - (x_mouse - xlim[0]) * scale,
            x_mouse + (xlim[1] - x_mouse) * scale,
        )
        new_ylim = (
            y_mouse - (y_mouse - ylim[0]) * scale,
            y_mouse + (ylim[1] - y_mouse) * scale,
        )
        for ax in self.image_axes:
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def selected_roi_xy(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.stats is None or self.selected_roi_idx is None:
            return None
        if not (0 <= self.selected_roi_idx < len(self.stats)):
            return None
        coords = self.roi_mask_coords(self.stats[self.selected_roi_idx], self.selected_mask_kind)
        if coords is None:
            return None
        z, y, x = coords
        if self.project_checkbox.isChecked() and not self.is_recording_selected():
            keep = np.ones(z.shape, dtype=bool)
        else:
            keep = z == self.plane_slider.value()
        if not np.any(keep):
            return None
        return x[keep], y[keep]

    def zoom_to_selected_roi(self) -> None:
        if not self.image_axes:
            return
        xy = self.selected_roi_xy()
        if xy is None:
            return
        x, y = xy
        margin = 35
        new_xlim = (float(x.min()) - margin, float(x.max()) + margin)
        new_ylim = (float(y.max()) + margin, float(y.min()) - margin)
        for ax in self.image_axes:
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def show_selected_roi_3d(self) -> None:
        if self.stats is None:
            QMessageBox.information(self, "No ROI masks", "No stats.npy file is loaded.")
            return
        if self.selected_roi_idx is None:
            QMessageBox.information(self, "No ROI selected", "Click an ROI mask first.")
            return
        if not (0 <= self.selected_roi_idx < len(self.stats)):
            QMessageBox.information(self, "Invalid ROI", f"ROI {self.selected_roi_idx} is not valid.")
            return

        stat = self.stats[self.selected_roi_idx]
        mask_kind = self.selected_mask_kind
        if mask_kind == "neuropil" and self.roi_mask_coords(stat, "neuropil") is None:
            QMessageBox.information(
                self,
                "No neuropil coordinates",
                f"ROI {self.selected_roi_idx} has no saved npcoords neuropil mask.",
            )
            return
        rois_dir = self.info_path.parent if self.info_path is not None else None
        window = Roi3DWindow(self.selected_roi_idx, stat, rois_dir=rois_dir, parent=self, mask_kind=mask_kind)
        window.finished.connect(lambda _result, w=window: self.forget_roi_3d_window(w))
        self.roi_3d_windows.append(window)
        window.show()

    def forget_roi_3d_window(self, window: Roi3DWindow) -> None:
        if window in self.roi_3d_windows:
            self.roi_3d_windows.remove(window)

    def reset_zoom(self) -> None:
        if (self.current_array is None and not self.is_recording_selected()) or not self.image_axes:
            return
        image, _ = self.current_image()
        ny, nx = image.shape
        for ax in self.image_axes:
            ax.set_xlim(-0.5, nx - 0.5)
            ax.set_ylim(ny - 0.5, -0.5)
        self.canvas.draw_idle()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a GUI for Suite3D rois/info.npy files.")
    parser.add_argument(
        "info",
        nargs="?",
        type=Path,
        default=None,
        help="Optional path to a Suite3D run folder, rois folder, or info.npy file.",
    )
    parser.add_argument(
        "--no-default",
        action="store_true",
        help="Deprecated; the viewer now opens empty unless a path is provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    viewer = InfoViewer(args.info)
    viewer.show()
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
