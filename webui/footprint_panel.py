"""
Footprint overlay panel for the suite3d web UI.

Shows detected cell footprints overlaid on the reference image or correlation map,
with interactive sliders to filter cells by various quality metrics.
"""
import numpy as np
import os
from pathlib import Path

import panel as pn
import param
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearColorMapper, Slider, RangeSlider, Select
from bokeh.palettes import Greys256
from bokeh.layouts import column, row

from suite3d.utils import load_iscell, make_iscell, normalize_iscell


class FootprintPanel(param.Parameterized):
    def __init__(self, max_height=None):
        super().__init__()
        self.job = None
        self.stats = None
        self.iscell = None
        self.vmap = None
        self.ref_img = None
        self.mean_img = None
        self.nz = 1

        # Feature arrays for filtering
        self.features = {}
        self.filter_ranges = {}

        # Bokeh sources
        self.bg_source = ColumnDataSource(data=dict(image=[np.zeros((10, 10))], dw=[10], dh=[10]))
        self.overlay_source = ColumnDataSource(data=dict(
            image=[np.zeros((10, 10), dtype=np.uint32)], dw=[10], dh=[10],
        ))

        # Build the layout
        self._build_ui(max_height)

    def _build_ui(self, max_height):
        """Construct the panel layout with plot and controls."""
        # Main figure
        self.plot = figure(
            height=550, width=550, title="Cell Footprints",
            tools="pan,reset,save,wheel_zoom",
            aspect_ratio=1, match_aspect=True,
            active_scroll="wheel_zoom", sizing_mode="scale_both",
        )
        self.plot.grid.grid_line_width = 0

        self.bg_mapper = LinearColorMapper(palette=Greys256, low=0, high=1)
        self.plot.image(
            source=self.bg_source, x=0, y=0, dw="dw", dh="dh",
            color_mapper=self.bg_mapper, image="image",
        )
        # Overlay with RGBA
        self.plot.image_rgba(
            source=self.overlay_source, x=0, y=0, dw="dw", dh="dh",
            image="image",
        )

        # Controls
        self.bg_select = pn.widgets.Select(
            name="Background", options=["Correlation Map", "Reference Image", "Mean Image"],
            value="Correlation Map", width=200,
        )
        self.z_slider = pn.widgets.IntSlider(name="Z Plane", start=0, end=1, step=1, value=0, width=200)
        self.contrast_slider = pn.widgets.RangeSlider(
            name="Contrast", start=0.0, end=1.0, step=0.01, value=(0.0, 1.0), width=200,
        )
        self.opacity_slider = pn.widgets.FloatSlider(
            name="Footprint Opacity", start=0.0, end=1.0, step=0.05, value=0.5, width=200,
        )

        # Filter sliders - will be populated when data loads
        self.filter_widgets = {}
        self.filter_pane = pn.Column(
            pn.pane.Markdown("### Cell Filters"),
            sizing_mode="stretch_width",
        )
        self.n_cells_display = pn.pane.Markdown("**Cells shown: 0 / 0**")

        # Bind callbacks
        pn.bind(self._on_z_change, self.z_slider, watch=True)
        pn.bind(self._on_bg_change, self.bg_select, watch=True)
        pn.bind(self._on_contrast_change, self.contrast_slider, watch=True)
        pn.bind(self._on_opacity_change, self.opacity_slider, watch=True)

        # Assemble layout
        controls = pn.Column(
            self.bg_select,
            self.z_slider,
            self.contrast_slider,
            self.opacity_slider,
            pn.layout.Divider(),
            self.n_cells_display,
            self.filter_pane,
            width=250,
        )
        plot_pane = pn.pane.Bokeh(self.plot, sizing_mode="scale_both")

        self.layout = pn.Row(
            plot_pane,
            controls,
            name="Footprints",
            sizing_mode="scale_both",
            max_height=max_height,
        )

    def load_job(self, job_interface):
        """Load segmentation data from the current job."""
        self.job = job_interface.job
        jobdir = job_interface.job_data["jobdir"]
        summary = job_interface.job_data["summary"]

        # Load reference and mean images
        self.ref_img = summary.get("ref_img_3d")

        # Load corrmap results
        try:
            corr_results = self.job.load_corr_map_results()
            self.vmap = corr_results.get("vmap")
            self.mean_img = corr_results.get("mean_img")
        except Exception:
            self.vmap = None
            self.mean_img = None

        # Load segmentation results
        try:
            rois_dir = self.job.dirs.get("rois")
            if rois_dir is None:
                print("No rois directory found")
                return
            stats_path = os.path.join(rois_dir, "stats.npy")
            if not os.path.exists(stats_path):
                print("No stats.npy found")
                return
            self.stats = np.load(stats_path, allow_pickle=True)

            iscell_path = os.path.join(rois_dir, "iscell.npy")
            if os.path.exists(iscell_path):
                self.iscell = load_iscell(iscell_path)
            else:
                self.iscell = make_iscell(len(self.stats))

        except Exception as e:
            print(f"Could not load segmentation results: {e}")
            return

        # Determine volume shape from best available source
        if self.vmap is not None:
            self.nz, self.ny, self.nx = self.vmap.shape
        elif self.ref_img is not None:
            self.nz, self.ny, self.nx = self.ref_img.shape
        else:
            print("No volume data available")
            return

        # Compute features for filtering
        self._compute_features()

        # Update UI
        self.z_slider.end = self.nz - 1
        self.z_slider.value = min(self.nz // 2, self.nz - 1)

        self._build_filter_sliders()
        self._update_display()

    def _compute_features(self):
        """Compute per-ROI features for filtering sliders."""
        self.features = {}
        if self.stats is None:
            return

        n_voxels = []
        peak_vals = []
        thresholds = []

        for stat in self.stats:
            n_voxels.append(len(stat.get("lam", [])))
            peak_vals.append(stat.get("peak_val", 0.0))
            thresholds.append(stat.get("threshold", 0.0))

        self.features["n_voxels"] = np.array(n_voxels, dtype=float)
        self.features["peak_val"] = np.array(peak_vals, dtype=float)
        self.features["threshold"] = np.array(thresholds, dtype=float)

        # Initialize ranges to full extent
        for key, vals in self.features.items():
            if len(vals) > 0:
                self.filter_ranges[key] = (float(np.nanmin(vals)), float(np.nanmax(vals)))

    def _build_filter_sliders(self):
        """Create range sliders for each feature."""
        self.filter_widgets = {}
        slider_list = [pn.pane.Markdown("### Cell Filters")]

        display_names = {
            "n_voxels": "# Voxels",
            "peak_val": "Corrmap Peak",
            "threshold": "Activity Threshold",
        }

        for key, vals in self.features.items():
            if len(vals) == 0:
                continue
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
            if vmax <= vmin:
                vmax = vmin + 1.0
            step = max((vmax - vmin) / 200, 0.001)

            slider = pn.widgets.RangeSlider(
                name=display_names.get(key, key),
                start=vmin, end=vmax, step=step,
                value=(vmin, vmax),
                width=220,
            )
            pn.bind(self._on_filter_change, slider, watch=True)
            self.filter_widgets[key] = slider
            slider_list.append(slider)

        self.filter_pane.objects = slider_list

    def _get_filtered_mask(self):
        """Return a boolean mask of which ROIs pass all current filters."""
        if self.stats is None:
            return np.array([], dtype=bool)

        n_roi = len(self.stats)
        mask = self.iscell.copy() if self.iscell is not None else np.ones(n_roi, dtype=bool)

        for key, slider in self.filter_widgets.items():
            vals = self.features.get(key)
            if vals is None:
                continue
            lo, hi = slider.value
            mask &= (vals >= lo) & (vals <= hi)

        return mask

    def _get_background(self, z):
        """Return the background image for the given z-plane."""
        bg_type = self.bg_select.value
        if bg_type == "Correlation Map" and self.vmap is not None:
            return self.vmap[z]
        elif bg_type == "Reference Image" and self.ref_img is not None:
            return self.ref_img[z]
        elif bg_type == "Mean Image" and self.mean_img is not None:
            return self.mean_img[z]
        # Fallback
        for vol in [self.vmap, self.ref_img, self.mean_img]:
            if vol is not None:
                return vol[z]
        return np.zeros((self.ny, self.nx))

    def _make_overlay(self, z, mask):
        """Create an RGBA overlay image for the given z-plane and ROI mask."""
        overlay = np.zeros((self.ny, self.nx, 4), dtype=np.uint8)
        if self.stats is None:
            return overlay

        opacity = int(self.opacity_slider.value * 255)

        # Color cycle for distinguishing cells
        colors = [
            (144, 190, 109),  # green
            (233, 138, 21),   # orange
            (178, 108, 152),  # purple
            (27, 154, 170),   # teal
            (58, 64, 90),     # dark blue
            (255, 99, 71),    # tomato
            (100, 149, 237),  # cornflower
            (255, 215, 0),    # gold
        ]

        cell_idx = 0
        for i, stat in enumerate(self.stats):
            if not mask[i]:
                continue
            coords = stat.get("coords")
            lam = stat.get("lam")
            if coords is None or lam is None:
                continue
            cz, cy, cx = coords
            # Only draw voxels on this z-plane
            z_mask = cz == z
            if not z_mask.any():
                continue

            color = colors[cell_idx % len(colors)]
            lam_z = lam[z_mask]
            cy_z = cy[z_mask]
            cx_z = cx[z_mask]

            # Normalize lambda for alpha blending
            lam_norm = lam_z / lam_z.max() if lam_z.max() > 0 else lam_z

            # Clip to valid image bounds
            valid = (cy_z >= 0) & (cy_z < self.ny) & (cx_z >= 0) & (cx_z < self.nx)
            cy_z, cx_z, lam_norm = cy_z[valid], cx_z[valid], lam_norm[valid]

            overlay[cy_z, cx_z, 0] = color[0]
            overlay[cy_z, cx_z, 1] = color[1]
            overlay[cy_z, cx_z, 2] = color[2]
            overlay[cy_z, cx_z, 3] = (lam_norm * opacity).astype(np.uint8)

            cell_idx += 1

        return overlay

    def _update_display(self):
        """Redraw background and overlay for current settings."""
        z = self.z_slider.value
        bg = self._get_background(z)

        # Update background
        lo, hi = self.contrast_slider.value
        bg_min, bg_max = float(np.nanmin(bg)), float(np.nanmax(bg))
        # Map slider [0,1] to actual data range
        actual_lo = bg_min + lo * (bg_max - bg_min)
        actual_hi = bg_min + hi * (bg_max - bg_min)

        self.bg_source.data.update(image=[bg], dw=[self.nx], dh=[self.ny])
        self.bg_mapper.low = actual_lo
        self.bg_mapper.high = actual_hi

        # Update overlay
        mask = self._get_filtered_mask()
        overlay = self._make_overlay(z, mask)

        # Convert RGBA to uint32 for image_rgba
        overlay_uint32 = np.zeros((self.ny, self.nx), dtype=np.uint32)
        view = overlay_uint32.view(dtype=np.uint8).reshape((self.ny, self.nx, 4))
        view[:] = overlay

        self.overlay_source.data.update(image=[overlay_uint32], dw=[self.nx], dh=[self.ny])

        # Update cell count
        total = int(mask.shape[0]) if mask.shape[0] > 0 else 0
        shown = int(mask.sum())
        self.n_cells_display.object = f"**Cells shown: {shown} / {total}**"

    # Callbacks
    def _on_z_change(self, value):
        self._update_display()

    def _on_bg_change(self, value):
        # Reset contrast to full range when switching backgrounds
        self.contrast_slider.value = (0.0, 1.0)
        self._update_display()

    def _on_contrast_change(self, value):
        z = self.z_slider.value
        bg = self._get_background(z)
        lo, hi = value
        bg_min, bg_max = float(np.nanmin(bg)), float(np.nanmax(bg))
        self.bg_mapper.low = bg_min + lo * (bg_max - bg_min)
        self.bg_mapper.high = bg_min + hi * (bg_max - bg_min)

    def _on_opacity_change(self, value):
        self._update_display()

    def _on_filter_change(self, value):
        self._update_display()
