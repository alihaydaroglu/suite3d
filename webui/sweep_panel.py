"""
Segmentation sweep panel for the suite3d web UI.

Shows results of segmentation parameter sweeps with interactive comparison
of ROI counts, size distributions, duplication indices, and shot noise.
Designed to update incrementally as sweep combinations complete.
"""
import numpy as np
import os
import glob
from pathlib import Path

import panel as pn
import param
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearColorMapper, HoverTool, Span
from bokeh.palettes import Greys256, Category10_10
from bokeh.layouts import column, row
from bokeh.transform import dodge

from suite3d import quality_metrics as qm


class SweepPanel(param.Parameterized):
    def __init__(self, max_height=None):
        super().__init__()
        self.job = None
        self.sweep_summary = None
        self.sweep_dir = None
        self.all_metrics = []
        self.all_stats = []
        self.param_names = []
        self.combinations = []
        self.voxel_size_um = (1, 1, 1)
        self.frate_hz = 1.0

        # Bokeh sources for the 4 main plots
        self.roi_count_source = ColumnDataSource(data=dict(
            x=[], total=[], filtered=[], labels=[],
        ))
        self.dup_source = ColumnDataSource(data=dict(
            x=[], dup_count=[], dup_rate=[], labels=[],
        ))
        self.noise_source = ColumnDataSource(data=dict(
            x=[], median=[], q25=[], q75=[], labels=[],
        ))
        # Scatter source for duplication corr vs dist
        self.dup_scatter_source = ColumnDataSource(data=dict(
            dist=[], corr=[], combo=[],
        ))
        # Size distribution: one source per combo, stored in a list
        self.size_sources = []

        self._build_ui(max_height)

    def _build_ui(self, max_height):
        """Construct the panel layout."""

        # --- Sweep selector ---
        self.sweep_select = pn.widgets.Select(
            name="Sweep", options=[], value=None, width=250,
        )
        self.refresh_btn = pn.widgets.Button(
            name="Refresh", button_type="primary", width=100,
        )
        self.min_npix_input = pn.widgets.IntInput(
            name="Min voxels filter", value=10, start=0, end=10000, step=1, width=120,
        )
        self.dup_thresh_input = pn.widgets.FloatInput(
            name="Dup corr threshold", value=0.8, start=0.0, end=1.0, step=0.05, width=120,
        )
        self.status_text = pn.pane.Markdown("**No sweep loaded**")
        self.summary_table = pn.pane.HTML("", sizing_mode="stretch_width")

        # --- Plots ---
        # 1. ROI count bar chart
        self.roi_plot = figure(
            height=300, width=400, title="ROI Counts",
            x_range=[], toolbar_location="above",
            tools="pan,reset,save,wheel_zoom",
        )
        self.roi_plot.vbar(
            x=dodge('x', -0.15, range=self.roi_plot.x_range),
            top='total', width=0.3, source=self.roi_count_source,
            color=Category10_10[0], alpha=0.7, legend_label="Total",
        )
        self.roi_plot.vbar(
            x=dodge('x', 0.15, range=self.roi_plot.x_range),
            top='filtered', width=0.3, source=self.roi_count_source,
            color=Category10_10[1], alpha=0.7, legend_label="Filtered",
        )
        self.roi_plot.xaxis.major_label_orientation = 0.7
        self.roi_plot.legend.location = "top_left"
        self.roi_plot.legend.label_text_font_size = "8pt"

        # 2. Duplication bar chart
        self.dup_plot = figure(
            height=300, width=400, title="Duplicate Pairs",
            x_range=[], toolbar_location="above",
            tools="pan,reset,save,wheel_zoom",
        )
        self.dup_plot.vbar(
            x='x', top='dup_count', width=0.6, source=self.dup_source,
            color=Category10_10[3], alpha=0.7,
        )
        self.dup_plot.xaxis.major_label_orientation = 0.7

        # 3. Duplication scatter: corr vs distance
        self.dup_scatter_plot = figure(
            height=300, width=400, title="Duplication: Corr vs Distance",
            x_axis_label="Distance (um)", y_axis_label="Pairwise correlation",
            tools="pan,reset,save,wheel_zoom",
        )
        self.dup_scatter_plot.scatter(
            'dist', 'corr', source=self.dup_scatter_source,
            size=2, alpha=0.15, color=Category10_10[3],
        )
        dup_line = Span(location=0.8, dimension='width',
                        line_color='red', line_dash='dashed', line_width=1)
        self.dup_scatter_plot.add_layout(dup_line)

        # 4. Shot noise bar chart
        self.noise_plot = figure(
            height=300, width=400, title="Median Shot Noise",
            x_range=[], toolbar_location="above",
            tools="pan,reset,save,wheel_zoom",
        )
        self.noise_plot.vbar(
            x='x', top='median', width=0.6, source=self.noise_source,
            color=Category10_10[2], alpha=0.7,
        )
        self.noise_plot.xaxis.major_label_orientation = 0.7

        # --- Bind callbacks ---
        pn.bind(self._on_sweep_change, self.sweep_select, watch=True)
        self.refresh_btn.on_click(self._on_refresh)
        pn.bind(self._on_filter_change, self.min_npix_input, watch=True)
        pn.bind(self._on_filter_change, self.dup_thresh_input, watch=True)

        # --- Assemble layout ---
        controls = pn.Column(
            pn.pane.Markdown("### Sweep Controls"),
            self.sweep_select,
            self.refresh_btn,
            pn.layout.Divider(),
            self.min_npix_input,
            self.dup_thresh_input,
            pn.layout.Divider(),
            self.status_text,
            width=280,
        )

        top_row = pn.Row(
            pn.pane.Bokeh(self.roi_plot),
            pn.pane.Bokeh(self.dup_plot),
        )
        bottom_row = pn.Row(
            pn.pane.Bokeh(self.dup_scatter_plot),
            pn.pane.Bokeh(self.noise_plot),
        )

        plots = pn.Column(
            top_row,
            bottom_row,
            self.summary_table,
            sizing_mode="stretch_both",
        )

        self.layout = pn.Row(
            controls,
            plots,
            name="Extraction Sweeps",
            sizing_mode="stretch_both",
            max_height=max_height,
        )

    def load_job(self, job_interface):
        """Load available sweeps from the current job."""
        self.job = job_interface.job
        self.voxel_size_um = self.job.params.get('voxel_size_um', (1, 1, 1))
        self.frate_hz = self.job.params.get('fs', 1.0)

        # Find all sweep directories
        sweeps_parent = os.path.join(self.job.job_dir, 'sweeps')
        if not os.path.isdir(sweeps_parent):
            self.status_text.object = "**No sweeps/ directory found**"
            return

        sweep_names = sorted([
            d for d in os.listdir(sweeps_parent)
            if os.path.isdir(os.path.join(sweeps_parent, d))
        ])

        if not sweep_names:
            self.status_text.object = "**No sweeps found**"
            return

        self.sweep_select.options = sweep_names
        self.sweep_select.value = sweep_names[-1]  # most recent
        self._load_sweep(sweep_names[-1])

    def _load_sweep(self, sweep_name):
        """Load a sweep's summary and compute metrics for completed combinations."""
        if self.job is None:
            return

        self.sweep_dir = os.path.join(self.job.job_dir, 'sweeps', sweep_name)
        summary_path = os.path.join(self.sweep_dir, 'sweep_summary.npy')

        if not os.path.exists(summary_path):
            self.status_text.object = f"**No sweep_summary.npy in {sweep_name}**"
            return

        self.sweep_summary = np.load(summary_path, allow_pickle=True).item()
        self.param_names = self.sweep_summary.get('param_names', [])
        self.combinations = self.sweep_summary.get('combinations', [])

        self._compute_metrics()
        self._update_plots()

    def _compute_metrics(self):
        """Compute quality metrics for all completed sweep combinations."""
        if self.sweep_summary is None:
            return

        results = self.sweep_summary.get('results', [])
        n_combs = len(self.combinations)
        n_done = len(results)

        self.all_metrics = []
        self.all_stats = []

        for i, res in enumerate(results):
            stats = res.get('stats', [])
            if isinstance(stats, dict):
                stats = stats.get('stats', [])
            self.all_stats.append(stats)

            # Try to load F from the combo's roi directory
            F = None
            roi_dir = res.get('roi_dir', '')
            if roi_dir and isinstance(roi_dir, str):
                F_path = os.path.join(roi_dir, 'F.npy')
                if os.path.exists(F_path):
                    try:
                        F = np.load(F_path)
                    except Exception:
                        pass

            if len(stats) == 0:
                self.all_metrics.append(None)
                continue

            m = qm.compute_roi_metrics(
                stats, F=F,
                voxel_size_um=self.voxel_size_um,
                frate_hz=self.frate_hz,
                near_thresh=20.0,
                min_npix=self.min_npix_input.value,
            )
            self.all_metrics.append(m)

        # Status
        self.status_text.object = (
            f"**{n_done} / {n_combs} combinations done**\n\n"
            f"Params: {', '.join(self.param_names)}"
        )

    def _update_plots(self):
        """Update all plots with current metrics."""
        if not self.all_metrics:
            return

        min_npix = self.min_npix_input.value
        dup_thresh = self.dup_thresh_input.value
        n = len(self.all_metrics)

        labels = []
        x_vals = []
        totals = []
        filtered_counts = []
        dup_counts = []
        dup_rates = []
        noise_meds = []
        noise_q25s = []
        noise_q75s = []
        all_dists = []
        all_corrs = []
        all_combos = []

        for i in range(n):
            combo = self.combinations[i]
            label = ", ".join(f"{self.param_names[j]}={combo[j]}"
                              for j in range(len(self.param_names)))
            short_label = ", ".join(f"{combo[j]}" for j in range(len(self.param_names)))

            m = self.all_metrics[i]
            if m is None:
                labels.append(short_label)
                x_vals.append(short_label)
                totals.append(0)
                filtered_counts.append(0)
                dup_counts.append(0)
                dup_rates.append(0)
                noise_meds.append(0)
                noise_q25s.append(0)
                noise_q75s.append(0)
                continue

            nv = m['n_voxels']
            dc = m['duplicate_corrs']
            dd = m['duplicate_dists']
            shot = m['shot_noise']

            n_total = m['n_rois']
            n_filt = int((nv >= min_npix).sum())
            n_dup = int((dc > dup_thresh).sum()) if len(dc) > 0 else 0

            valid_noise = shot[~np.isnan(shot)]
            med_noise = float(np.median(valid_noise)) if len(valid_noise) > 0 else 0
            q25 = float(np.percentile(valid_noise, 25)) if len(valid_noise) > 0 else 0
            q75 = float(np.percentile(valid_noise, 75)) if len(valid_noise) > 0 else 0

            labels.append(label)
            x_vals.append(short_label)
            totals.append(n_total)
            filtered_counts.append(n_filt)
            dup_counts.append(n_dup)
            dup_rates.append(n_dup / max(n_total, 1) * 100)
            noise_meds.append(med_noise)
            noise_q25s.append(q25)
            noise_q75s.append(q75)

            # Scatter data (subsample if too many)
            if len(dc) > 0:
                max_pts = 2000
                if len(dc) > max_pts:
                    idx = np.random.choice(len(dc), max_pts, replace=False)
                    all_dists.extend(dd[idx].tolist())
                    all_corrs.extend(dc[idx].tolist())
                else:
                    all_dists.extend(dd.tolist())
                    all_corrs.extend(dc.tolist())
                all_combos.extend([short_label] * min(len(dc), max_pts))

        # Update ROI count plot
        self.roi_plot.x_range.factors = x_vals
        self.roi_count_source.data = dict(
            x=x_vals, total=totals, filtered=filtered_counts, labels=labels,
        )

        # Update dup plot
        self.dup_plot.x_range.factors = x_vals
        self.dup_plot.title.text = f"Duplicate Pairs (corr > {dup_thresh})"
        self.dup_source.data = dict(
            x=x_vals, dup_count=dup_counts, dup_rate=dup_rates, labels=labels,
        )

        # Update dup scatter
        self.dup_scatter_source.data = dict(
            dist=all_dists, corr=all_corrs, combo=all_combos,
        )
        # Update threshold line
        for r in self.dup_scatter_plot.renderers:
            if hasattr(r, 'location') and hasattr(r, 'dimension'):
                r.location = dup_thresh

        # Update noise plot
        self.noise_plot.x_range.factors = x_vals
        self.noise_source.data = dict(
            x=x_vals, median=noise_meds, q25=noise_q25s, q75=noise_q75s, labels=labels,
        )

        # Summary HTML table
        rows = []
        for i in range(n):
            m = self.all_metrics[i]
            if m is None:
                rows.append(f"<tr><td>C{i}</td><td>{labels[i]}</td>"
                            "<td colspan='4'>not completed</td></tr>")
                continue
            nv = m['n_voxels']
            dc = m['duplicate_corrs']
            n_dup = int((dc > dup_thresh).sum()) if len(dc) > 0 else -1
            valid_noise = m['shot_noise'][~np.isnan(m['shot_noise'])]
            med_n = f"{np.median(valid_noise):.4f}" if len(valid_noise) > 0 else "N/A"
            rows.append(
                f"<tr><td>C{i}</td><td>{labels[i]}</td>"
                f"<td>{m['n_rois']}</td>"
                f"<td>{int((nv >= min_npix).sum())}</td>"
                f"<td>{np.median(nv):.0f}</td>"
                f"<td>{n_dup}</td>"
                f"<td>{med_n}</td></tr>"
            )

        table_html = f"""
        <table style="border-collapse: collapse; font-size: 11px; width: 100%;">
        <thead><tr style="background: #f0f0f0;">
            <th style="padding: 4px; border: 1px solid #ddd;">#</th>
            <th style="padding: 4px; border: 1px solid #ddd;">Parameters</th>
            <th style="padding: 4px; border: 1px solid #ddd;">Total</th>
            <th style="padding: 4px; border: 1px solid #ddd;">&ge;{min_npix}vx</th>
            <th style="padding: 4px; border: 1px solid #ddd;">Med vx</th>
            <th style="padding: 4px; border: 1px solid #ddd;">Dups</th>
            <th style="padding: 4px; border: 1px solid #ddd;">Noise</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
        </table>
        """
        self.summary_table.object = table_html

    # --- Callbacks ---

    def _on_sweep_change(self, value):
        if value:
            self._load_sweep(value)

    def _on_refresh(self, event):
        """Re-read sweep summary from disk (picks up newly completed combinations)."""
        if self.sweep_select.value:
            self._load_sweep(self.sweep_select.value)

    def _on_filter_change(self, value):
        """Re-compute metrics with new filter thresholds."""
        self._compute_metrics()
        self._update_plots()
