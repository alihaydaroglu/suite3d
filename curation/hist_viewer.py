import numpy as np
import panel as pn
import h5py
from bokeh.plotting import figure
from bokeh.models import Span, ColumnDataSource, BoxAnnotation
from bokeh.palettes import Blues8, Greens8, Reds8
from bokeh.layouts import gridplot

class HistViewer:
    """Component for viewing histograms of data properties with population, cluster, and individual sample comparisons"""
    
    def __init__(self, dataset, properties, sample_indices=None, use_sampling=False):
        self.dataset = dataset
        self.properties = properties
        self.sample_indices = sample_indices
        self.use_sampling = use_sampling
        
        # Cache management
        self.population_cache = None  # Population histograms
        self.cluster_cache = {}       # Per-cluster histograms
        self.current_cluster_id = None
        self.current_individual_properties = None
        
        # Histogram properties to compute
        self.property_names = [k for k, v in self.properties.items() if v is not None and len(np.unique(v)) > 1]
        
        # UI Controls
        self.property_selector = pn.widgets.Select(
            name="Property to Display",
            options=self.property_names,
            value=self.property_names[0],
            width=200
        )
        
        self.n_bins_slider = pn.widgets.IntSlider(
            name="Number of Bins",
            start=20, end=100, value=50,
            width=200
        )
        
        # Status text
        self.status_text = pn.pane.Markdown("**Status:** Initializing...", width=200)
        
        # Threshold state
        self._current_threshold_mask = None
        self.threshold_box = None  # Will be created with plot

        # Histogram plot
        self.hist_plot = self._create_empty_plot()
        self.hist_sources = {
            'population': ColumnDataSource(data=dict(top=[], left=[], right=[])),
            'cluster': ColumnDataSource(data=dict(top=[], left=[], right=[])),
        }
        self.individual_span = None  # Track the current span to remove old ones
        
        # Set up callbacks
        self.property_selector.param.watch(self._on_property_change, 'value')
        self.n_bins_slider.param.watch(self._on_bins_change, 'value')
        
        # Callback for individual sample changes (set by BoxViewer)
        self.on_individual_sample_changed = None

        # Threshold classification widgets
        self.threshold_mode = pn.widgets.RadioButtonGroup(
            name="Threshold Mode",
            options=["Off", "Below", "Above", "Range"],
            value="Off",
            width=200
        )

        self.threshold_slider = pn.widgets.FloatSlider(
            name="Threshold",
            start=0, end=1, value=0.5,
            step=0.01,
            width=200,
            visible=False
        )

        self.threshold_range = pn.widgets.RangeSlider(
            name="Range",
            start=0, end=1, value=(0.25, 0.75),
            step=0.01,
            width=200,
            visible=False
        )

        self.threshold_preview = pn.pane.Markdown("", width=200)

        self.apply_cell_btn = pn.widgets.Button(
            name="CELL",
            button_type='danger',
            width=95,
            visible=False
        )

        self.apply_not_cell_btn = pn.widgets.Button(
            name="NOT CELL",
            button_type='danger',
            button_style='outline',
            width=95,
            visible=False
        )

        # Threshold callbacks
        self.threshold_mode.param.watch(self._on_threshold_mode_change, 'value')
        self.threshold_slider.param.watch(self._on_threshold_change, 'value')
        self.threshold_range.param.watch(self._on_threshold_change, 'value')
        self.apply_cell_btn.on_click(self._on_apply_cell_click)
        self.apply_not_cell_btn.on_click(self._on_apply_not_cell_click)

        # Callback for threshold classification (set by AppOrchestrator)
        self.on_threshold_classify = None

        # Initialize
        if self.dataset is not None:
            self._compute_population_histograms()
    
    def _create_empty_plot(self):
        """Create empty bokeh plot for histograms"""
        plot = figure(
            width=700, height=350,  # Adjusted width for side-by-side layout with controls
            title="Property Distribution",
            toolbar_location="above",
            x_axis_label="Property Value",
            y_axis_label="Probability Density"
        )
        plot.title.text_font_size = "12pt"
        plot.yaxis.ticker.desired_num_ticks = 0

        self.threshold_box = BoxAnnotation(
            fill_alpha=0.3,
            fill_color='orange',
            line_color='orange',
            line_width=2,
            line_dash='dashed',
            visible=False
        )
        plot.add_layout(self.threshold_box)

        return plot
    
    def _compute_sample_properties(self, sample_indices):
        """Compute all histogram properties for a single sample or batch of samples
        
        Args:
            sample_data: List of indices to use to get the sample data (indexing into self.dataset, the full unsampled dataset)
            
        Returns:
            dict with property_name -> value(s)
        """
        if len(sample_indices) == 1:                                           # Single sample
            sample_data = self.dataset[sample_indices[0]]
            sample_data = sample_data[np.newaxis, ...]  # Add batch dimension
        else:
            sample_data = self.dataset[sample_indices]            
        
        props = {}
        for key, value in self.properties.items():
            if value is not None:
                if key == 'Edge Cells':
                    # Special case: Edge Cells is boolean mask
                    props[key] = value[sample_indices].astype(np.float32)
                else:
                    props[key] = value[sample_indices]
        
        return props

    def _compute_population_histograms(self):
        """Compute histograms for the entire population (or sample)"""
        if self.dataset is None:
            return
        
        self.status_text.object = "**Status:** Computing population histograms..."
        
        try:
            # Use same sampling strategy as other components
            if self.use_sampling and self.sample_indices is not None:
                indices_to_use = np.sort(self.sample_indices)  # Sort indices for HDF5
                n_samples = len(self.sample_indices)
                sampling_msg = f" (sampled from {self.dataset.shape[0]:,})"
            else:
                n_samples = min(10000, self.dataset.shape[0])  # Limit for performance
                indices_to_use = np.sort(np.random.choice(self.dataset.shape[0], n_samples, replace=False))
                sampling_msg = f" (random sample of {n_samples:,})"
            
            # Load data in chunks to manage memory
            chunk_size = 5000
            all_properties = {prop: [] for prop in self.property_names}

            for i in range(0, len(indices_to_use), chunk_size):
                chunk_indices = indices_to_use[i:i+chunk_size]
                chunk_properties = self._compute_sample_properties(chunk_indices)
                
                for prop_name in self.property_names:
                    all_properties[prop_name].extend(chunk_properties[prop_name])
            
            # Convert to numpy arrays and compute normalized histograms (PDFs)
            self.population_cache = {}
            for prop_name in self.property_names:
                values = np.array(all_properties[prop_name])
                hist, bin_edges = np.histogram(values, bins=self.n_bins_slider.value)
                
                # Normalize to PDF: divide by (total_count * bin_width)
                bin_width = bin_edges[1] - bin_edges[0]
                pdf_hist = hist / (np.sum(hist) * bin_width)
                
                self.population_cache[prop_name] = {
                    'hist': pdf_hist,
                    'bin_edges': bin_edges,
                    'values': values
                }
            
            self.status_text.object = f"**Status:** Population histograms ready ({n_samples:,} samples{sampling_msg})"
            self._update_plot()
            
        except Exception as e:
            self.status_text.object = f"**Error:** Failed to compute population histograms: {str(e)}"
    
    def _compute_single_property_histogram(self, prop_name):
        """Compute histogram for a single property (used when adding properties dynamically)"""
        if self.dataset is None or self.properties.get(prop_name) is None:
            return

        try:
            if prop_name != 'Probability':
                # Use same sampling strategy as population histograms
                if self.use_sampling and self.sample_indices is not None:
                    indices_to_use = self.sample_indices
                else:
                    indices_to_use = np.arange(min(10000, self.dataset.shape[0]))

                values = self.properties[prop_name][indices_to_use]
            else:
                values = self.properties[prop_name]
            hist, bin_edges = np.histogram(values, bins=self.n_bins_slider.value)

            # Normalize to PDF
            bin_width = bin_edges[1] - bin_edges[0]
            pdf_hist = hist / (np.sum(hist) * bin_width) if np.sum(hist) > 0 else hist

            if self.population_cache is None:
                self.population_cache = {}

            self.population_cache[prop_name] = {
                'hist': pdf_hist,
                'bin_edges': bin_edges,
                'values': values
            }

            self._update_plot()
        except Exception as e:
            self.status_text.object = f"**Error:** Failed to compute histogram for {prop_name}: {str(e)}"

    def load_cluster_data(self, cluster_id, display_data):
        """Load and cache cluster histograms"""
        if self.dataset is None:
            return
            
        # Check cache
        if cluster_id in self.cluster_cache:
            self.current_cluster_id = cluster_id
            self.status_text.object = f"**Status:** Using cached cluster {cluster_id} histograms"
            self._update_plot()
            return
        
        try:
            # Get cluster indices (same pattern as BoxViewer)
            cluster_mask = display_data['cluster'] == cluster_id
            cluster_data = display_data[cluster_mask]
            cluster_original_indices = cluster_data['original_index'].values
            
            if len(cluster_original_indices) == 0:
                self.status_text.object = f"**Error:** No points in cluster {cluster_id}"
                return
            
            self.status_text.object = f"**Status:** Computing histograms for cluster {cluster_id} ({len(cluster_original_indices)} samples)..."
            
            # Sort indices for HDF5 access - we don't care about preserving order for histograms
            sorted_indices = np.sort(cluster_original_indices)
            
            # Compute properties
            cluster_properties = self._compute_sample_properties(sorted_indices)
            
            # Compute histograms using same bins as population, normalized to PDFs
            cluster_hist_data = {}
            for prop_name in self.property_names:
                if self.population_cache and prop_name in self.population_cache:
                    # Use same bin edges as population for comparison
                    bin_edges = self.population_cache[prop_name]['bin_edges']
                    hist, _ = np.histogram(cluster_properties[prop_name], bins=bin_edges)
                else:
                    # Fallback to auto-binning
                    hist, bin_edges = np.histogram(cluster_properties[prop_name], bins=self.n_bins_slider.value)
                
                # Normalize to PDF: divide by (total_count * bin_width)
                bin_width = bin_edges[1] - bin_edges[0]
                pdf_hist = hist / (np.sum(hist) * bin_width) if np.sum(hist) > 0 else hist
                
                cluster_hist_data[prop_name] = {
                    'hist': pdf_hist,
                    'bin_edges': bin_edges,
                    'values': cluster_properties[prop_name]
                }
            
            self.cluster_cache[cluster_id] = cluster_hist_data
            self.current_cluster_id = cluster_id
            
            self.status_text.object = f"**Status:** Cluster {cluster_id} histograms ready"
            self._update_plot()
            
        except Exception as e:
            self.status_text.object = f"**Error:** Failed to load cluster data: {str(e)}"
    
    def update_individual_sample(self, sample_data):
        """Update individual sample marker from current BoxViewer sample"""
        if sample_data is None:
            self.current_individual_properties = None
            self._update_plot()
            return
        
        try:
            # Compute properties for the individual sample
            individual_props = self._compute_sample_properties(sample_data)
            self.current_individual_properties = individual_props
            self._update_plot()
            
        except Exception as e:
            self.status_text.object = f"**Error:** Failed to process individual sample: {str(e)}"
    
    def remove_property(self, name):
        """Dynamically remove a property option"""
        if name in self.property_names:
            self.property_names.remove(name)
            self.property_selector.options = self.property_names
            self.property_selector.param.trigger('options')
            # Reset to first property if current selection was removed
            if self.property_selector.value == name and self.property_names:
                self.property_selector.value = self.property_names[0]
        if name in self.properties:
            self.properties[name] = None
        if self.population_cache and name in self.population_cache:
            del self.population_cache[name]
        self._update_plot()

    def update_property(self, name, values):
        """Update an existing property with new values from the app orchestrator

        Args:
            name: Name of the property to update
            values: New values for the property (array-like)
        """
        self.properties[name] = values

        # Add to property names if not already present
        if name not in self.property_names:
            self.property_names.append(name)
            self.property_selector.options = self.property_names
            self.property_selector.param.trigger('options')

        # Clear all cached cluster histograms since property values changed
        # We clear the entire cache because load_cluster_data checks if cluster_id
        # is in cache and returns early - partial cache would cause missing histograms
        self.cluster_cache = {}

        # Recompute the population histogram for this property
        self._compute_single_property_histogram(name)
    
    def _update_plot(self):
        """Update the histogram plot based on current selections"""
        selected_property = self.property_selector.value
        
        # Remove existing individual span if it exists
        if self.individual_span is not None:
            try:
                for l in [self.hist_plot.center, self.hist_plot.left, self.hist_plot.right, self.hist_plot.above, self.hist_plot.below]:
                    if self.individual_span in l:
                        l.remove(self.individual_span)
            except (ValueError, AttributeError):
                pass  # Span was already removed or doesn't exist

            self.individual_span = None
        
        # Clear existing renderers
        self.hist_plot.renderers = []
        
        try:
            if (hasattr(self.hist_plot, 'legend') and 
                self.hist_plot.legend is not None and 
                hasattr(self.hist_plot.legend, 'items') and 
                len(self.hist_plot.legend.items) > 0):
                self.hist_plot.legend.items = []
        except (AttributeError, TypeError):
            pass
        
        # Plot population histogram
        if self.population_cache and selected_property in self.population_cache:
            pop_data = self.population_cache[selected_property]
            hist = pop_data['hist']
            bin_edges = pop_data['bin_edges']
            
            # Create bar coordinates
            left_edges = bin_edges[:-1]
            right_edges = bin_edges[1:]
            
            self.hist_sources['population'].data = dict(
                top=hist,
                left=left_edges,
                right=right_edges
            )
            
            self.hist_plot.quad(
                top='top', bottom=0, left='left', right='right',
                source=self.hist_sources['population'],
                alpha=0.3, color=Blues8[2], legend_label="Population"
            )
        
        # Plot cluster histogram
        if (self.current_cluster_id is not None and self.current_cluster_id in self.cluster_cache 
            and selected_property in self.cluster_cache[self.current_cluster_id]):
            
            cluster_data = self.cluster_cache[self.current_cluster_id][selected_property]
            hist = cluster_data['hist']
            bin_edges = cluster_data['bin_edges']
            
            left_edges = bin_edges[:-1]
            right_edges = bin_edges[1:]
            
            self.hist_sources['cluster'].data = dict(
                top=hist,
                left=left_edges,
                right=right_edges
            )
            
            self.hist_plot.quad(
                top='top', bottom=0, left='left', right='right',
                source=self.hist_sources['cluster'],
                alpha=0.6, color=Greens8[4], legend_label=f"Cluster {self.current_cluster_id}"
            )
        
        # Plot individual sample marker
        if (self.current_individual_properties is not None and 
            selected_property in self.current_individual_properties):
            
            individual_value = self.current_individual_properties[selected_property]
            if hasattr(individual_value, '__len__') and len(individual_value) > 0:
                individual_value = individual_value[0]  # Take first if array
            
            self.individual_span = Span(
                location=individual_value, 
                dimension='height', 
                line_color=Reds8[1], 
                line_width=2,
                line_dash='dashed'
            )
            self.hist_plot.add_layout(self.individual_span)
        
        # Update plot title and labels
        self.hist_plot.title.text = f"Distribution: {selected_property}"
        self.hist_plot.xaxis.axis_label = selected_property
        
        # Update legend
        self.hist_plot.legend.location = "top_right"
        self.hist_plot.legend.click_policy = "hide"
    
    def _on_property_change(self, event):
        """Handle property selector changes"""
        self._update_plot()
        # Update threshold bounds when property changes
        if self.threshold_mode.value != "Off":
            self._update_threshold_bounds()
            self._update_threshold_visualization()
    
    def _on_display_change(self, event):
        """Handle display option changes"""
        self._update_plot()
    
    def _on_bins_change(self, event):
        """Handle bins slider changes - requires recomputing histograms"""
        # Clear caches to force recomputation with new bin count
        self.population_cache = None
        self.cluster_cache = {}
        self.current_cluster_id = None
        
        # Recompute
        if self.dataset is not None:
            self._compute_population_histograms()
    
    def clear_cache(self):
        """Clear all caches (call when clustering changes)"""
        self.cluster_cache = {}
        self.current_cluster_id = None
        self.current_individual_properties = None
        self.status_text.object = "**Status:** Cache cleared"
        self._update_plot()

    def _on_threshold_mode_change(self, event):
        """Handle threshold mode changes - show/hide appropriate widgets"""
        mode = event.new

        # Show/hide single vs range slider
        self.threshold_slider.visible = mode in ["Above", "Below"]
        self.threshold_range.visible = mode == "Range"

        # Show/hide action buttons
        show_buttons = mode != "Off"
        self.apply_cell_btn.visible = show_buttons
        self.apply_not_cell_btn.visible = show_buttons

        # Update slider ranges based on current property
        if mode != "Off":
            self._update_threshold_bounds()

        # Update visualization
        self._update_threshold_visualization()

    def _update_threshold_bounds(self):
        """Update threshold slider bounds based on current property data"""
        selected_property = self.property_selector.value
        if self.population_cache and selected_property in self.population_cache:
            values = self.population_cache[selected_property]['values']
            data_min, data_max = float(np.nanmin(values)), float(np.nanmax(values))

            # Add small margin
            margin = (data_max - data_min) * 0.02

            # Update single slider
            self.threshold_slider.start = data_min - margin
            self.threshold_slider.end = data_max + margin
            self.threshold_slider.value = (data_min + data_max) / 2
            self.threshold_slider.step = (data_max - data_min) / 100

            # Update range slider
            self.threshold_range.start = data_min - margin
            self.threshold_range.end = data_max + margin
            self.threshold_range.value = (
                data_min + (data_max - data_min) * 0.25,
                data_min + (data_max - data_min) * 0.75
            )
            self.threshold_range.step = (data_max - data_min) / 100

    def _on_threshold_change(self, event):
        """Handle threshold slider value changes"""
        self._update_threshold_visualization()

    def _update_threshold_visualization(self):
        """Update BoxAnnotation and preview count based on current threshold"""
        mode = self.threshold_mode.value

        if mode == "Off" or not self.population_cache:
            self.threshold_box.visible = False
            self.threshold_preview.object = ""
            self._current_threshold_mask = None
            return

        selected_property = self.property_selector.value
        if selected_property not in self.population_cache:
            self.threshold_box.visible = False
            self.threshold_preview.object = ""
            self._current_threshold_mask = None
            return

        # Get property values
        values = self.population_cache[selected_property]['values']

        # Calculate affected indices based on mode
        if mode == "Below":
            threshold = self.threshold_slider.value
            mask = values < threshold
            self.threshold_box.left = None  # Extends to left edge
            self.threshold_box.right = threshold
        elif mode == "Above":
            threshold = self.threshold_slider.value
            mask = values > threshold
            self.threshold_box.left = threshold
            self.threshold_box.right = None  # Extends to right edge
        elif mode == "Range":
            low, high = self.threshold_range.value
            mask = (values >= low) & (values <= high)
            self.threshold_box.left = low
            self.threshold_box.right = high

        self.threshold_box.visible = True
        self._current_threshold_mask = mask

        # Calculate preview counts
        count = int(np.sum(mask))
        total = len(values)
        percentage = (count / total) * 100 if total > 0 else 0

        self.threshold_preview.object = f"**Preview:** {count:,} points ({percentage:.1f}%)"

    def get_threshold_mask(self):
        """Return boolean mask of points matching current threshold (for orchestrator)"""
        return self._current_threshold_mask

    def _on_apply_cell_click(self, event):
        """Handle 'Apply as CELL' button click"""
        if self.on_threshold_classify:
            self.on_threshold_classify('cell')

    def _on_apply_not_cell_click(self, event):
        """Handle 'Apply as NOT CELL' button click"""
        if self.on_threshold_classify:
            self.on_threshold_classify('not_cell')

    def get_layout(self):
        """Return the complete HistViewer layout"""
        # Main controls in a vertical layout
        controls = pn.Column(
            self.property_selector,
            pn.Spacer(height=10),
            self.n_bins_slider,
            "### Threshold Classification",
            self.threshold_mode,
            self.threshold_slider,
            self.threshold_range,
            self.threshold_preview,
            pn.Row(self.apply_cell_btn, self.apply_not_cell_btn, width=200),
            self.status_text,
            width=240,
            margin=(10, 10)
        )

        plot_pane = pn.pane.Bokeh(self.hist_plot, sizing_mode='fixed', height=350, width=700)

        return pn.Row(
            pn.Spacer(width=20),
            plot_pane,
            pn.Spacer(width=30),
            controls,
            sizing_mode='stretch_width',
            margin=(10, 0)
        )
    
    def __del__(self):
        """Clean up HDF5 file handle"""
        if self.hdf5_file is not None:
            try:
                self.hdf5_file.close()
            except:
                pass


