"""
Segmentation sweep panel for the suite3d web UI.

Shows results of segmentation parameter sweeps with interactive comparison
of ROI counts, size distributions, duplication indices, and shot noise.
Designed to update incrementally as sweep combinations complete.

Performance notes:
- Basic metrics (ROI count, sizes) shown instantly from sweep_summary
- Duplication + shot noise require F.npy (computed on load, cached to disk)
- Overmerge requires movie data (triggered by button, cached to disk)
"""
import numpy as np
import os
import hashlib
import threading
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
        self.mov_subset = None

        # Bokeh sources
        self.roi_count_source = ColumnDataSource(data=dict(
            x=[], total=[], filtered=[], labels=[],
        ))
        self.dup_source = ColumnDataSource(data=dict(
            x=[], dup_count=[], dup_rate=[], labels=[],
        ))
        self.noise_source = ColumnDataSource(data=dict(
            x=[], median=[], q25=[], q75=[], labels=[],
        ))
        self.overmerge_source = ColumnDataSource(data=dict(
            x=[], median=[], pct_above=[], labels=[],
        ))
        self.dup_scatter_source = ColumnDataSource(data=dict(
            dist=[], corr=[], combo=[],
        ))

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
        self.compute_overmerge_btn = pn.widgets.Button(
            name="Compute Overmerge", button_type="warning", width=150,
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

        self.overmerge_plot = figure(
            height=300, width=400, title="Overmerge Score (click button to compute)",
            x_range=[], toolbar_location="above",
            tools="pan,reset,save,wheel_zoom",
        )
        self.overmerge_plot.vbar(
            x='x', top='median', width=0.6, source=self.overmerge_source,
            color=Category10_10[4], alpha=0.7,
        )
        self.overmerge_plot.xaxis.major_label_orientation = 0.7
        self.overmerge_plot.yaxis.axis_label = "Median overmerge score"

        # --- Bind callbacks ---
        pn.bind(self._on_sweep_change, self.sweep_select, watch=True)
        self.refresh_btn.on_click(self._on_refresh)
        self.compute_overmerge_btn.on_click(self._on_compute_overmerge)
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
            self.compute_overmerge_btn,
            pn.layout.Divider(),
            self.status_text,
            width=280,
        )

        top_row = pn.Row(
            pn.pane.Bokeh(self.roi_plot),
            pn.pane.Bokeh(self.dup_plot),
        )
        mid_row = pn.Row(
            pn.pane.Bokeh(self.dup_scatter_plot),
            pn.pane.Bokeh(self.noise_plot),
        )
        bottom_row = pn.Row(
            pn.pane.Bokeh(self.overmerge_plot),
        )

        plots = pn.Column(
            top_row,
            mid_row,
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

    # ---- Data loading ----

    def load_job(self, job_interface):
        """Load available sweeps from the current job. No movie loading here."""
        self.job = job_interface.job
        self.voxel_size_um = self.job.params.get('voxel_size_um', (1, 1, 1))
        self.frate_hz = self.job.params.get('fs', 1.0)
        self.mov_subset = None  # loaded lazily on overmerge button

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
        self.sweep_select.value = sweep_names[-1]
        self._load_sweep(sweep_names[-1])

    def _load_sweep(self, sweep_name):
        """Load sweep summary and compute fast metrics (no overmerge)."""
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

        # Try to load cached metrics first
        cache_path = os.path.join(self.sweep_dir, 'sweep_metrics_cache.npy')
        if os.path.exists(cache_path):
            try:
                cached = np.load(cache_path, allow_pickle=True).item()
                if cached.get('n_results') == len(self.sweep_summary.get('results', [])):
                    self.all_metrics = cached['metrics']
                    self.all_stats = cached.get('stats_refs', [None] * len(self.all_metrics))
                    self._update_plots()
                    n_done = len(self.all_metrics)
                    n_combs = len(self.combinations)
                    self.status_text.object = (
                        f"**{n_done} / {n_combs} done (cached)**\n\n"
                        f"Params: {', '.join(self.param_names)}"
                    )
                    return
            except Exception:
                pass  # cache invalid, recompute

        self._compute_metrics(include_overmerge=False)
        self._update_plots()

    def _compute_metrics(self, include_overmerge=False):
        """Compute quality metrics for completed sweep combinations.

        Args:
            include_overmerge: If True, loads movie subset and computes
                PCA-based overmerge scores (slow). Otherwise skips overmerge.
        """
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

            # Try to load F
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

            mov = self.mov_subset if include_overmerge else None
            m = qm.compute_roi_metrics(
                stats, F=F,
                voxel_size_um=self.voxel_size_um,
                frate_hz=self.frate_hz,
                mov=mov,
                near_thresh=20.0,
                min_npix=self.min_npix_input.value,
            )
            self.all_metrics.append(m)

        # Cache to disk
        self._save_metrics_cache()

        self.status_text.object = (
            f"**{n_done} / {n_combs} combinations done**\n\n"
            f"Params: {', '.join(self.param_names)}"
        )

    def _save_metrics_cache(self):
        """Save computed metrics to disk for fast reload."""
        if self.sweep_dir is None:
            return
        try:
            cache = {
                'n_results': len(self.sweep_summary.get('results', [])),
                'metrics': self.all_metrics,
            }
            cache_path = os.path.join(self.sweep_dir, 'sweep_metrics_cache.npy')
            np.save(cache_path, cache, allow_pickle=True)
        except Exception as e:
            print(f"  Could not save metrics cache: {e}")

    def _ensure_movie_loaded(self):
        """Lazily load movie subset for overmerge computation."""
        if self.mov_subset is not None:
            return True
        if self.job is None:
            return False

        reg_dir = self.job.dirs.get(
            'registered_fused_data',
            os.path.join(self.job.job_dir, 'registered_fused_data')
        )
        if not os.path.isdir(reg_dir):
            return False

        self.status_text.object = "**Loading movie subset for overmerge...**"
        try:
            self.mov_subset = qm.load_movie_subset(reg_dir, max_frames=300)
            if self.mov_subset is not None:
                print(f"  Loaded movie subset: {self.mov_subset.shape}")
                return True
        except Exception as e:
            print(f"  Could not load movie: {e}")
        return False

    # ---- Plot updates ----

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
        om_meds = []
        om_pct_above = []
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
                om_meds.append(0)
                om_pct_above.append(0)
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

            # Overmerge
            om = m['overmerge_scores']
            valid_om = om[~np.isnan(om)]
            om_meds.append(float(np.median(valid_om)) if len(valid_om) > 0 else 0)
            om_pct_above.append(float((valid_om > 0.5).sum() / max(len(valid_om), 1) * 100)
                                if len(valid_om) > 0 else 0)

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

        # Update all bokeh sources
        self.roi_plot.x_range.factors = x_vals
        self.roi_count_source.data = dict(
            x=x_vals, total=totals, filtered=filtered_counts, labels=labels,
        )

        self.dup_plot.x_range.factors = x_vals
        self.dup_plot.title.text = f"Duplicate Pairs (corr > {dup_thresh})"
        self.dup_source.data = dict(
            x=x_vals, dup_count=dup_counts, dup_rate=dup_rates, labels=labels,
        )

        self.dup_scatter_source.data = dict(
            dist=all_dists, corr=all_corrs, combo=all_combos,
        )

        self.noise_plot.x_range.factors = x_vals
        self.noise_source.data = dict(
            x=x_vals, median=noise_meds, q25=noise_q25s, q75=noise_q75s, labels=labels,
        )

        # Overmerge
        self.overmerge_plot.x_range.factors = x_vals
        has_om = any(v > 0 for v in om_meds)
        if has_om:
            self.overmerge_plot.title.text = "Median Overmerge Score"
        else:
            self.overmerge_plot.title.text = "Overmerge Score (click button to compute)"
        self.overmerge_source.data = dict(
            x=x_vals, median=om_meds, pct_above=om_pct_above, labels=labels,
        )

        # Summary HTML table
        rows = []
        for i in range(n):
            m = self.all_metrics[i]
            if m is None:
                rows.append(f"<tr><td>C{i}</td><td>{labels[i]}</td>"
                            "<td colspan='5'>not completed</td></tr>")
                continue
            nv = m['n_voxels']
            dc = m['duplicate_corrs']
            n_dup = int((dc > dup_thresh).sum()) if len(dc) > 0 else -1
            valid_noise = m['shot_noise'][~np.isnan(m['shot_noise'])]
            med_n = f"{np.median(valid_noise):.4f}" if len(valid_noise) > 0 else "N/A"
            om = m['overmerge_scores']
            valid_om = om[~np.isnan(om)]
            med_om = f"{np.median(valid_om):.3f}" if len(valid_om) > 0 else "N/A"
            rows.append(
                f"<tr><td>C{i}</td><td>{labels[i]}</td>"
                f"<td>{m['n_rois']}</td>"
                f"<td>{int((nv >= min_npix).sum())}</td>"
                f"<td>{np.median(nv):.0f}</td>"
                f"<td>{n_dup}</td>"
                f"<td>{med_n}</td>"
                f"<td>{med_om}</td></tr>"
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
            <th style="padding: 4px; border: 1px solid #ddd;">Overmerge</th>
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
        self._compute_metrics(include_overmerge=False)
        self._update_plots()

    def _on_compute_overmerge(self, event):
        """Load movie and compute overmerge scores (triggered by button)."""
        self.status_text.object = "**Loading movie for overmerge...**"
        self.compute_overmerge_btn.disabled = True

        def _do_overmerge():
            try:
                if not self._ensure_movie_loaded():
                    self.status_text.object = "**Could not load movie for overmerge**"
                    self.compute_overmerge_btn.disabled = False
                    return
                self.status_text.object = "**Computing overmerge scores...**"
                self._compute_metrics(include_overmerge=True)
                self._update_plots()
                self.status_text.object = "**Overmerge computed and cached**"
            except Exception as e:
                self.status_text.object = f"**Overmerge error:** {e}"
            finally:
                self.compute_overmerge_btn.disabled = False

        # Run in thread to not block UI
        thread = threading.Thread(target=_do_overmerge, daemon=True)
        thread.start()
