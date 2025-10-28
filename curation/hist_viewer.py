import numpy as np
import panel as pn
import h5py
from bokeh.plotting import figure
from bokeh.models import Span, ColumnDataSource
from bokeh.palettes import Blues8, Greens8, Reds8
from bokeh.layouts import gridplot

class HistViewer:
    """Component for viewing histograms of data properties with population, cluster, and individual sample comparisons"""
    
    def __init__(self, hdf5_path, sample_indices=None, use_sampling=False):
        self.hdf5_path = hdf5_path
        self.sample_indices = sample_indices
        self.use_sampling = use_sampling
        self.hdf5_file = None
        self.dataset = None
        
        # Cache management
        self.population_cache = None  # Population histograms
        self.cluster_cache = {}       # Per-cluster histograms
        self.current_cluster_id = None
        self.current_individual_properties = None
        
        # Histogram properties to compute
        self.property_names = [
            'Mean Intensity Ch1', 'Mean Intensity Ch2', 'Mean Intensity Ch3'
        ]
        
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
        
        # Initialize
        self._open_hdf5()
        if self.dataset is not None:
            self._compute_population_histograms()
    
    def _open_hdf5(self):
        """Open HDF5 file for reading (same pattern as BoxViewer)"""
        try:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')
            
            # Find the main dataset (same logic as BoxViewer)
            if len(self.hdf5_file.keys()) == 1:
                dataset_key = list(self.hdf5_file.keys())[0]
                self.dataset = self.hdf5_file[dataset_key]
                self.status_text.object = f"**Status:** Opened HDF5 dataset '{dataset_key}'"
            else:
                possible_keys = ['data', 'dataset', 'samples', 'X']
                for key in possible_keys:
                    if key in self.hdf5_file:
                        self.dataset = self.hdf5_file[key]
                        self.status_text.object = f"**Status:** Using dataset '{key}'"
                        break
                else:
                    available_keys = list(self.hdf5_file.keys())
                    self.status_text.object = f"**Error:** Please specify dataset. Available: {available_keys}"
                    return
            
            # Verify expected shape
            if self.dataset.ndim != 5 or self.dataset.shape[1:] != (3, 5, 20, 20):
                self.status_text.object = f"**Error:** Unexpected data shape: {self.dataset.shape}"
                self.dataset = None
                
        except Exception as e:
            self.status_text.object = f"**Error:** Could not open HDF5 file: {str(e)}"
    
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
        return plot
    
    def _compute_sample_properties(self, sample_data):
        """Compute all histogram properties for a single sample or batch of samples
        
        Args:
            sample_data: Either (3, 5, 20, 20) for single sample or (N, 3, 5, 20, 20) for batch
            
        Returns:
            dict with property_name -> value(s)
        """
        if sample_data.ndim == 4:  # Single sample
            sample_data = sample_data[np.newaxis, ...]  # Add batch dimension
        
        batch_size = sample_data.shape[0]
        properties = {}
        
        for ch in range(3):
            channel_data = sample_data[:, ch, :, :, :]
            
            # Mean intensity per sample
            properties[f'Mean Intensity Ch{ch+1}'] = np.mean(channel_data, axis=(1, 2, 3))
            
            # Max intensity per sample  
            properties[f'Max Intensity Ch{ch+1}'] = np.max(channel_data, axis=(1, 2, 3))
            
            # Z-centroid (weighted average Z position)
            z_weights = np.sum(channel_data, axis=(2, 3))  # Sum over X, Y: (batch_size, 5)
            z_positions = np.arange(5)[np.newaxis, :]  # (1, 5)
            z_centroids = np.sum(z_weights * z_positions, axis=1) / (np.sum(z_weights, axis=1) + 1e-8)
            properties[f'Z-Centroid Ch{ch+1}'] = z_centroids
            
            if ch == 0:  # Only compute spatial properties for first channel to avoid redundancy
                # Center of mass for X and Y (using max projection)
                max_proj = np.max(channel_data, axis=1)  # (batch_size, 20, 20)
                
                # Create coordinate grids
                y_coords, x_coords = np.meshgrid(np.arange(20), np.arange(20), indexing='ij')
                
                # Compute center of mass for each sample
                com_x = np.zeros(batch_size)
                com_y = np.zeros(batch_size)
                
                for i in range(batch_size):
                    total_intensity = np.sum(max_proj[i]) + 1e-8
                    com_x[i] = np.sum(max_proj[i] * x_coords) / total_intensity
                    com_y[i] = np.sum(max_proj[i] * y_coords) / total_intensity
                
                properties['Center of Mass X Ch1'] = com_x
                properties['Center of Mass Y Ch1'] = com_y
        
        return properties
    
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
            chunk_size = 1000
            all_properties = {prop: [] for prop in self.property_names}
            
            for i in range(0, len(indices_to_use), chunk_size):
                chunk_indices = indices_to_use[i:i+chunk_size]
                chunk_data = self.dataset[chunk_indices]
                chunk_properties = self._compute_sample_properties(chunk_data)
                
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
            
            # Sort indices for HDF5 access, then unsort results (same pattern as BoxViewer)
            sort_order = np.argsort(cluster_original_indices)
            sorted_indices = cluster_original_indices[sort_order]
            cluster_samples_sorted = self.dataset[sorted_indices]
            
            # Restore original order
            unsort_order = np.argsort(sort_order)
            cluster_samples = cluster_samples_sorted[unsort_order]
            
            # Compute properties
            cluster_properties = self._compute_sample_properties(cluster_samples)
            
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
    
    def get_layout(self):
        """Return the complete HistViewer layout"""
        # Main controls in a vertical layout
        controls = pn.Column(
            "### Histogram Viewer",
            self.property_selector,
            pn.Spacer(height=10),
            self.n_bins_slider,
            pn.Spacer(height=10),
            self.status_text,
            width=240,
            margin=(10, 10)
        )
        
        plot_pane = pn.pane.Bokeh(self.hist_plot, sizing_mode='fixed', height=350)
        
        return pn.Row(
            plot_pane,
            pn.Spacer(width=20),
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


