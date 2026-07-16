from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QToolTip,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QKeySequence

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


DEFAULT_INFO_PATH = Path(r"D:\suite3d_runs\SS010_2026-07-07\s3d-v1\rois\info.npy")
DISPLAY_KEYS = ("max_img", "mean_img", "vmap", "vmap_raw")
RECORDING_DISPLAY_KEY = "registered movie"
TRACE_FILES = {
    "F": "F.npy",
    "Fneu": "Fneu.npy",
    "spks": "spks.npy",
}
SURFACE_SMOOTHING_SIGMA = (0.35, 0.65, 0.65)
GUI_BG = "#4a4a4a"
PANEL_BG = "#5a5a5a"
PLOT_BG = "#6a6a6a"
TEXT_FG = "#f2f2f2"
GRID_FG = "#d8d8d8"


class Roi3DWindow(QDialog):
    def __init__(self, roi_idx: int, stat: dict, rois_dir: Path | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"ROI {roi_idx} 3D view")
        self.roi_idx = roi_idx
        self.stat = stat
        self.rois_dir = rois_dir
        self.correlation_cache: np.ndarray | None = None

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Render"))
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItems(["All ROI voxels", "Smoothed surface"])
        self.render_mode_combo.currentTextChanged.connect(self.plot_roi)
        controls.addWidget(self.render_mode_combo)
        controls.addSpacing(16)
        controls.addWidget(QLabel("Color"))
        self.color_mode_combo = QComboBox()
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
        coords = self.stat.get("coords")
        lam = self.stat.get("lam")
        if coords is None or len(coords) != 3 or lam is None:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"ROI {self.roi_idx} has no 3D coordinates", ha="center", va="center")
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        z, y, x = [np.asarray(c, dtype=float) for c in coords]
        weights = np.asarray(lam, dtype=float)
        values = weights
        colorbar_label = "ROI's spatial weights"

        if self.color_mode_combo.currentText() == "Pixel correlation":
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
            mappable = self.plot_roi_points(ax, x, y, z, values, geometry_weights=weights)
        self.style_3d_axes(ax)
        ax.set_title(f"{self.roi_title(self.roi_idx, self.stat, weights.size)}; {render_mode}", color=TEXT_FG)
        self.set_3d_limits(ax, x, y, z)
        if mappable is not None:
            cbar = self.figure.colorbar(mappable, ax=ax, label=colorbar_label, shrink=0.75)
            cbar.ax.yaxis.label.set_color(TEXT_FG)
            cbar.ax.tick_params(colors=TEXT_FG)
        self.canvas.draw_idle()

    def roi_title(self, roi_idx: int, stat: dict, n_voxels: int) -> str:
        return f"ROI {roi_idx}; {n_voxels} voxels"

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
            return self.plot_roi_points(ax, x, y, z, values, geometry_weights=geometry_weights)

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
                return self.plot_roi_points(ax, x, y, z, values, geometry_weights=geometry_weights)

            level = max(float(smoothed.max()) * 0.18, float(np.percentile(positive, 45)))
            verts, faces, _normals, vertex_geometry_values = marching_cubes(smoothed, level=level)
            vertex_color_values = self.sample_volume_nearest(smoothed_colors, verts)
            verts[:, 0] += origin[0]
            verts[:, 1] += origin[1]
            verts[:, 2] += origin[2]

            polygons = verts[faces][:, :, [2, 1, 0]]
            finite_color_values = vertex_color_values[np.isfinite(vertex_color_values)]
            if finite_color_values.size == 0 or float(finite_color_values.max()) <= float(finite_color_values.min()):
                finite_color_values = vertex_geometry_values[np.isfinite(vertex_geometry_values)]
            if finite_color_values.size == 0 or float(finite_color_values.max()) <= float(finite_color_values.min()):
                finite_color_values = positive

            norm = Normalize(vmin=float(finite_color_values.min()), vmax=float(finite_color_values.max()))
            face_values = vertex_color_values[faces].mean(axis=1)
            face_values = np.nan_to_num(face_values, nan=float(finite_color_values.min()))
            colors = cm.magma(norm(face_values))
            colors[:, 3] = 0.82

            surface = Poly3DCollection(polygons, facecolors=colors, linewidths=0.05, edgecolors=(1, 1, 1, 0.12))
            ax.add_collection3d(surface)
            center_weights = np.maximum(np.asarray(geometry_values, dtype=float), 0)
            if not np.any(center_weights > 0):
                center_weights = np.ones_like(center_weights)
            ax.scatter(
                [float(np.average(x, weights=center_weights))],
                [float(np.average(y, weights=center_weights))],
                [float(np.average(z, weights=center_weights))],
                color="white",
                s=18,
                alpha=0.85,
            )

            mappable = cm.ScalarMappable(norm=norm, cmap="magma")
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
            norm = Normalize(vmin=level, vmax=float(volume.max()))
            facecolors = cm.magma(norm(volume))
            facecolors[..., 3] = 0.55
            ax.voxels(x_idx, y_idx, z_idx, filled, facecolors=facecolors, edgecolor=(1, 1, 1, 0.08))
            mappable = cm.ScalarMappable(norm=norm, cmap="magma")
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

        return ax.scatter(
            x,
            y,
            z,
            c=plot_values,
            cmap="magma",
            norm=norm,
            s=sizes,
            alpha=0.9,
            edgecolors=(1, 1, 1, 0.18),
            linewidths=0.15,
            depthshade=False,
        )

    def style_3d_axes(self, ax) -> None:
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        ax.set_zlabel("z plane")
        ax.xaxis.label.set_color(TEXT_FG)
        ax.yaxis.label.set_color(TEXT_FG)
        ax.zaxis.label.set_color(TEXT_FG)
        ax.tick_params(colors=TEXT_FG)

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
        z_margin = max(float(z.max() - z.min()) * 0.1, 0.5)

        ax.set_xlim(float(x.min()) - x_margin, float(x.max()) + x_margin)
        ax.set_ylim(float(y.min()) - y_margin, float(y.max()) + y_margin)
        z_low = max(0.0, float(z.min()) - z_margin)
        z_high = float(z.max()) + z_margin
        ax.set_zlim(z_high, z_low)
        try:
            ax.set_box_aspect(
                (
                    max(float(x.max() - x.min()), 1.0),
                    max(float(y.max() - y.min()), 1.0),
                    max(float(z.max() - z.min()), 1.0),
                )
            )
        except AttributeError:
            pass


class InfoViewer(QMainWindow):
    def __init__(self, info_path: Path | None = None, load_default: bool = True) -> None:
        super().__init__()
        self.setWindowTitle("Suite3D info.npy viewer")

        self.info_path: Path | None = None
        self.info: dict | None = None
        self.current_array: np.ndarray | None = None
        self.current_image_shape: tuple[int, int, int] | None = None
        self.stats: np.ndarray | None = None
        self.iscell: np.ndarray | None = None
        self.roi_colors: np.ndarray | None = None
        self.roi_metrics: dict[str, np.ndarray] = {}
        self.curation_controls: dict[str, dict[str, object]] = {}
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
        self.selected_roi_idx: int | None = None
        self.selected_panel_accepted = True
        self.manual_curation_overrides: dict[int, bool] = {}
        self.traces: dict[str, np.ndarray] = {}
        self.trace_paths: dict[str, Path] = {}
        self.trace_roi_axes: dict[str, int] = {}
        self.trace_cursor_lines: list[object] = []
        self.motion_shifts: np.ndarray | None = None
        self.motion_shift_paths: list[Path] = []
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

        self._build_ui()

        if info_path is not None:
            self.load_info(info_path)
        elif load_default and DEFAULT_INFO_PATH.exists():
            self.load_info(DEFAULT_INFO_PATH)

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
            QPushButton, QComboBox {{
                background-color: #6b6b6b;
                color: {TEXT_FG};
                border: 1px solid #909090;
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QPushButton:hover, QComboBox:hover {{
                background-color: #777777;
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
        self.file_label = QLabel("No file loaded")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        file_row.addWidget(self.open_button)
        file_row.addWidget(self.open_file_button)
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

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["gray", "magma", "viridis", "inferno"])
        self.cmap_combo.currentTextChanged.connect(self.on_display_changed)
        controls_layout.addWidget(QLabel("Colormap"), 1, 0)
        controls_layout.addWidget(self.cmap_combo, 1, 1)

        self.masks_checkbox = QCheckBox("Show cell ROIs")
        self.masks_checkbox.stateChanged.connect(self.on_display_changed)
        controls_layout.addWidget(self.masks_checkbox, 1, 2)

        self.motion_checkbox = QCheckBox("Motion correction")
        self.motion_checkbox.stateChanged.connect(self.on_motion_correction_toggled)
        self.motion_checkbox.setEnabled(False)
        controls_layout.addWidget(self.motion_checkbox, 1, 3, 1, 2)

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
        controls_layout.addWidget(self.plane_slider, 2, 2, 1, 3)

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
        controls_layout.addWidget(self.frame_slider, 3, 2, 1, 3)

        self.contrast_label = QLabel("Contrast: -")
        self.contrast_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        controls_layout.addWidget(self.contrast_label, 4, 0, 1, 5)

        self.zoom_selected_button = QPushButton("Zoom selected ROI")
        self.zoom_selected_button.clicked.connect(self.zoom_to_selected_roi)
        controls_layout.addWidget(self.zoom_selected_button, 5, 0)

        self.reset_zoom_button = QPushButton("Reset zoom")
        self.reset_zoom_button.clicked.connect(self.reset_zoom)
        controls_layout.addWidget(self.reset_zoom_button, 5, 1)

        self.show_3d_button = QPushButton("Show selected ROI in 3D")
        self.show_3d_button.clicked.connect(self.show_selected_roi_3d)
        controls_layout.addWidget(self.show_3d_button, 5, 2)

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

        self.trace_group = QGroupBox("F trace")
        trace_layout = QVBoxLayout(self.trace_group)
        self.trace_figure = Figure(figsize=(12, 2.2), dpi=100, constrained_layout=True)
        self.trace_canvas = FigureCanvas(self.trace_figure)
        self.trace_canvas.setStyleSheet(f"background-color: {GUI_BG};")
        self.trace_canvas.setMinimumHeight(160)
        self.trace_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        trace_layout.addWidget(self.trace_canvas)
        root.addWidget(self.trace_group, stretch=1)
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

        self.resize(1500, 1000)

    def build_curation_panel(self) -> QGroupBox:
        group = QGroupBox("ROI curation")
        layout = QVBoxLayout(group)

        self.curation_summary_label = QLabel("No ROI masks loaded")
        layout.addWidget(self.curation_summary_label)

        self.curation_figure = Figure(figsize=(4.2, 5.0), dpi=100)
        self.curation_canvas = FigureCanvas(self.curation_figure)
        self.curation_canvas.setStyleSheet(f"background-color: {GUI_BG};")
        self.curation_canvas.setMinimumWidth(420)
        self.curation_canvas.setMinimumHeight(360)
        self.curation_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.curation_canvas.mpl_connect("button_press_event", self.on_curation_press)
        self.curation_canvas.mpl_connect("motion_notify_event", self.on_curation_motion)
        self.curation_canvas.mpl_connect("button_release_event", self.on_curation_release)
        self.curation_canvas.mpl_connect("figure_leave_event", self.on_curation_leave)
        layout.addWidget(self.curation_canvas, stretch=1)

        for key, label, decimals in self.curation_metric_definitions():
            self.curation_controls[key] = {
                "enabled": True,
                "min": 0.0,
                "max": 0.0,
                "range_min": -1e9,
                "range_max": 1e9,
                "label": label,
                "decimals": decimals,
            }

        self.reject_neuropil_button = QPushButton("Reject Fneu > F")
        self.reject_neuropil_button.clicked.connect(self.reject_high_neuropil_mean_rois)
        self.reject_neuropil_button.setEnabled(False)
        layout.addWidget(self.reject_neuropil_button)

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
        )

    def curation_metric_help(self) -> dict[str, str]:
        return {
            "npix": "ROI footprint size in voxels.",
            "zspan": "Number of z-planes touched by the ROI.",
            "peak_val": "Strength of the seed peak in the Suite3D detection map.",
            "vox_snr": "Median voxel signal-to-noise inside the ROI.",
        }

    def start_directory(self) -> Path:
        if self.info_path is not None:
            if self.info_path.parent.name == "rois":
                return self.info_path.parent.parent
            return self.info_path.parent
        if DEFAULT_INFO_PATH.exists():
            return DEFAULT_INFO_PATH.parent.parent
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
        self.roi_metrics = {}
        if hasattr(self, "curation_group"):
            self.curation_group.setEnabled(False)
            self.save_iscell_button.setEnabled(False)
            self.reject_neuropil_button.setEnabled(False)
            self.curation_summary_label.setText("No ROI masks loaded")
        self.current_id_maps = {}
        self.image_axes_panels = {}
        self.overlay_cache = {}
        self.selected_roi_idx = None
        self.selected_panel_accepted = True
        self.curation_undo_stack = []
        self.manual_curation_overrides = {}
        self.clear_selection_rectangle()
        self.traces = {}
        self.trace_paths = {}
        self.trace_roi_axes = {}

        stats_path = directory / "stats.npy"
        if not stats_path.exists():
            self.masks_checkbox.setChecked(False)
            self.masks_checkbox.setEnabled(False)
            self.clear_trace_plot("F trace: no stats.npy found")
            return

        try:
            self.stats = np.load(stats_path, allow_pickle=True)
        except Exception as exc:
            self.masks_checkbox.setChecked(False)
            self.masks_checkbox.setEnabled(False)
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
                values = self.roi_metrics.get(key)
                if controls is None or values is None:
                    continue
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    controls["enabled"] = False
                    continue
                min_value = float(np.nanmin(finite))
                max_value = float(np.nanmax(finite))
                margin = max((max_value - min_value) * 0.05, 1.0)
                controls["range_min"] = min_value - margin
                controls["range_max"] = max_value + margin
                controls["min"] = min_value
                controls["max"] = max_value
                controls["enabled"] = True
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
        }
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

    def apply_curation_filters(self) -> None:
        self.ensure_iscell_array()
        if self.iscell is None or not self.roi_metrics:
            return

        accepted = np.ones(self.iscell.shape[0], dtype=bool)
        for key, _label, _decimals in self.curation_metric_definitions():
            controls = self.curation_controls.get(key)
            values = self.roi_metrics.get(key)
            if controls is None or values is None or not controls["enabled"]:
                continue
            low = float(controls["min"])
            high = float(controls["max"])
            if low > high:
                low, high = high, low
            accepted &= np.isfinite(values) & (values >= low) & (values <= high)

        for roi_idx, manual_value in self.manual_curation_overrides.items():
            if 0 <= roi_idx < accepted.shape[0]:
                accepted[roi_idx] = bool(manual_value)

        self.iscell[:, 0] = accepted.astype(self.iscell.dtype)
        self.overlay_cache = {}
        self.update_curation_histograms()
        self.update_image(preserve_view=True)

    def update_curation_histograms(self) -> None:
        if not hasattr(self, "curation_figure"):
            return
        self.curation_figure.clear()
        self.curation_figure.patch.set_facecolor(GUI_BG)
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

        accepted_count = int((self.iscell[:, 0] > 0).sum()) if self.iscell is not None else 0
        total_count = len(self.stats)
        self.curation_summary_label.setText(
            f"Accepted: {accepted_count}    Non-accepted: {total_count - accepted_count}"
        )

        metric_defs = self.curation_metric_definitions()
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
            if controls is not None and controls["enabled"]:
                low = float(controls["min"])
                high = float(controls["max"])
                if low > high:
                    low, high = high, low
                low_line = ax.axvline(low, color="#d6d14a", linewidth=1.8, picker=6)
                high_line = ax.axvline(high, color="#d6d14a", linewidth=1.8, picker=6)
                self.curation_threshold_lines[(key, "min")] = low_line
                self.curation_threshold_lines[(key, "max")] = high_line
            title = ax.set_title(f"{label}: {finite.min():.4g} - {finite.max():.4g}", color=TEXT_FG, fontsize=8)
            self.curation_title_artists[title] = key
            ax.tick_params(colors=TEXT_FG, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(GRID_FG)

        self.curation_figure.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.07, hspace=0.75)
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
        if controls is None or not controls["enabled"]:
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

        self.apply_roi_acceptance(roi_indices, accepted=False)
        self.status_label.setText(f"Marked {roi_indices.size} ROIs as non-accepted because mean Fneu > mean F.")

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

        self.select_roi_by_index(int(roi_pool[pool_pos]), accepted_panel)

    def select_roi_by_index(self, roi_idx: int, accepted_panel: bool | None = None) -> None:
        if self.stats is None or not (0 <= roi_idx < len(self.stats)):
            return
        self.selected_roi_idx = roi_idx
        if accepted_panel is None:
            accepted_panel = bool(self.iscell is None or self.iscell[roi_idx, 0] > 0)
        self.selected_panel_accepted = bool(accepted_panel)
        self.update_trace_plot(roi_idx)
        self.update_image(preserve_view=True)

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

    def build_mask_overlay(
        self,
        image_shape: tuple[int, int],
        accepted_panel: bool,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        overlay, id_map = self.base_mask_overlay(image_shape, accepted_panel)
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
        )
        return selected_overlay, selected_id_map

    def base_mask_overlay(
        self,
        image_shape: tuple[int, int],
        accepted_panel: bool,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        if (
            self.stats is None
            or self.roi_colors is None
            or not self.masks_checkbox.isChecked()
        ):
            return None, np.full(image_shape, -1, dtype=np.int32)

        projected = self.project_checkbox.isChecked() and not self.is_recording_selected()
        plane = None if projected else int(self.plane_slider.value())
        cache_key = (tuple(image_shape), bool(accepted_panel), plane)
        cached = self.overlay_cache.get(cache_key)
        if cached is not None:
            return cached

        ny, nx = image_shape
        overlay = np.zeros((ny, nx, 4), dtype=np.float32)
        id_map = np.full((ny, nx), -1, dtype=np.int32)
        alpha = 0.55

        for roi_idx, stat in enumerate(self.stats):
            if not self.roi_matches_panel(roi_idx, accepted_panel):
                continue
            coords = stat.get("coords")
            if coords is None or len(coords) != 3:
                continue

            z, y, x = coords
            z = np.asarray(z)
            y = np.asarray(y)
            x = np.asarray(x)
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
            overlay[yy, xx, :3] = self.roi_colors[roi_idx]
            overlay[yy, xx, 3] = alpha
            id_map[yy, xx] = roi_idx

        if np.any(id_map >= 0):
            result = (overlay, id_map)
        else:
            result = (None, id_map)
        self.overlay_cache[cache_key] = result
        return result

    def add_selected_roi_to_overlay(
        self,
        overlay: np.ndarray | None,
        id_map: np.ndarray,
        image_shape: tuple[int, int],
        accepted_panel: bool,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        if self.stats is None or self.selected_roi_idx is None:
            return overlay, id_map
        if not self.roi_matches_panel(self.selected_roi_idx, accepted_panel):
            return overlay, id_map

        stat = self.stats[self.selected_roi_idx]
        coords = stat.get("coords")
        if coords is None or len(coords) != 3:
            return overlay, id_map

        ny, nx = image_shape
        z, y, x = [np.asarray(c) for c in coords]
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

        lo, hi = np.nanpercentile(image, [1, 99.8])
        if hi <= lo:
            lo = float(np.nanmin(image))
            hi = float(np.nanmax(image))
        self.contrast_label.setText(f"Contrast percentiles 1-99.8: {lo:.4g}, {hi:.4g}")

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

        ax_left = self.figure.add_subplot(1, 2, 1)
        ax_right = self.figure.add_subplot(1, 2, 2, sharex=ax_left, sharey=ax_left)
        self.current_axes = ax_left
        self.image_axes = [ax_left, ax_right]
        self.image_axes_panels = {ax_left: True, ax_right: False}

        panel_specs = [
            (ax_left, True, "Accepted cells"),
            (ax_right, False, "Non-accepted cells"),
        ]
        for ax, accepted_panel, panel_name in panel_specs:
            ax.set_facecolor(PLOT_BG)
            ax.imshow(image, cmap=self.cmap_combo.currentText(), vmin=lo, vmax=hi, aspect="equal")
            overlay, id_map = self.build_mask_overlay(image.shape, accepted_panel)
            self.current_id_maps[ax] = id_map
            if overlay is not None:
                ax.imshow(overlay, interpolation="nearest", aspect="equal")
            ax.set_title(panel_name, color=TEXT_FG)
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

    def clear_trace_plot(self, message: str) -> None:
        if not hasattr(self, "trace_figure"):
            return
        if hasattr(self, "trace_group") and not self.motion_checkbox.isChecked():
            self.trace_group.setTitle("F trace")
        self.trace_cursor_lines = []
        self.trace_figure.clear()
        self.trace_figure.patch.set_facecolor(GUI_BG)
        ax = self.trace_figure.add_subplot(111)
        ax.set_facecolor(PLOT_BG)
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color=TEXT_FG)
        ax.set_axis_off()
        self.trace_canvas.draw_idle()

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
            ax.set_ylabel(ylabel)
            ax.yaxis.label.set_color(TEXT_FG)
            ax.tick_params(colors=TEXT_FG)
            for spine in ax.spines.values():
                spine.set_color(GRID_FG)
            ax.grid(True, alpha=0.25, color=GRID_FG)
            legend = ax.legend(loc="upper right", facecolor=PLOT_BG, edgecolor=GRID_FG)
            for text in legend.get_texts():
                text.set_color(TEXT_FG)

        axes[0].set_title(
            f"Motion correction shifts; frame {frame}; "
            f"y={y_shift[frame]:.3g}, x={x_shift[frame]:.3g}, x-y={xy_motion[frame]:.3g} pixels",
            color=TEXT_FG,
        )
        axes[-1].set_xlabel("Frame")
        axes[-1].xaxis.label.set_color(TEXT_FG)
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
        bad = [key for key, value in available.items() if value.ndim != 1 or value.size == 0]
        if bad:
            self.clear_trace_plot(f"Trace: ROI {roi_idx} has invalid trace(s): {', '.join(bad)}")
            return

        self.trace_figure.clear()
        self.trace_figure.patch.set_facecolor(GUI_BG)
        self.trace_cursor_lines = []
        self.trace_group.setTitle("F trace")
        ax = self.trace_figure.add_subplot(111)
        ax.set_facecolor(PLOT_BG)

        if f_trace is not None:
            frames = np.arange(f_trace.size)
            ax.plot(frames, f_trace, color="lime", linewidth=0.8, label="F")
        if fneu_trace is not None:
            frames = np.arange(fneu_trace.size)
            ax.plot(frames, fneu_trace, color="red", linewidth=0.8, label="Neuropil")
        if spks_trace is not None:
            frames = np.arange(spks_trace.size)
            ax.plot(frames, spks_trace, color="white", linewidth=0.8, label="Deconvolved")

        cursor = ax.axvline(
            self.frame_slider.value(),
            color="#ffd54f",
            linewidth=1.5,
            linestyle="--",
            label="_nolegend_",
        )
        self.trace_cursor_lines = [cursor]

        ax.set_title(f"ROI {roi_idx} traces", color=TEXT_FG)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Signal")
        ax.xaxis.label.set_color(TEXT_FG)
        ax.yaxis.label.set_color(TEXT_FG)
        ax.tick_params(colors=TEXT_FG)
        for spine in ax.spines.values():
            spine.set_color(GRID_FG)
        ax.grid(True, alpha=0.25, color=GRID_FG)
        legend = ax.legend(loc="upper right", facecolor=PLOT_BG, edgecolor=GRID_FG)
        for text in legend.get_texts():
            text.set_color(TEXT_FG)
        if self.trace_paths:
            file_names = ", ".join(path.name for path in self.trace_paths.values())
            ax.text(
                0.995,
                0.96,
                file_names,
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=8,
                color=TEXT_FG,
            )
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
        self.overlay_cache = {}
        self.update_curation_histograms()
        self.update_image(preserve_view=True)

    def apply_roi_acceptance(self, roi_indices: np.ndarray, accepted: bool) -> None:
        if self.iscell is None or roi_indices.size == 0:
            return
        roi_indices = np.unique(roi_indices.astype(np.int64, copy=False))
        roi_indices = roi_indices[(roi_indices >= 0) & (roi_indices < self.iscell.shape[0])]
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
            if not self.motion_checkbox.isChecked():
                self.clear_trace_plot("F trace: no ROI selected")
            self.update_image(preserve_view=True)
            return

        self.select_roi_by_index(roi_idx, self.image_axes_panels.get(axes))

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
        coords = self.stats[self.selected_roi_idx].get("coords")
        if coords is None or len(coords) != 3:
            return None
        z, y, x = [np.asarray(c) for c in coords]
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
        rois_dir = self.info_path.parent if self.info_path is not None else None
        window = Roi3DWindow(self.selected_roi_idx, stat, rois_dir=rois_dir, parent=self)
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
        help="Open an empty viewer instead of loading the built-in default info.npy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    viewer = InfoViewer(args.info, load_default=not args.no_default)
    viewer.show()
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
