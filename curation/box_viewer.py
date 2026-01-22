import numpy as np
import panel as pn
from bokeh.plotting import figure
from bokeh.palettes import Greys256, Blues8

class BoxViewer:
    """Component for viewing 5D data samples with 3 channels and dual view projections"""
    
    def __init__(self, dataset, sample_indices=None, use_sampling=False):
        self.dataset = dataset
        self.sample_indices = sample_indices
        self.use_sampling = use_sampling
        self.display_dataset = None  # Subset of dataset to display (if sample_indices provided)
        self.cluster_cache = {}
        self.current_cluster_id = None
        self.current_sample = None  # Index in original (full) dataset of the datapoint to display. Will be a list, with length > 1 only if Mean option is selected
        
        # Keep track of image renderers to avoid recreation
        self.xy_image_renderers = [None, None, None]
        self.xz_image_renderers = [None, None, None]
        
        # UI Controls
        self.view_selector = pn.widgets.RadioButtonGroup(
            name="View Type",
            options=["Selected", "Mean", "Median"],
            value="Selected",
            width=200
        )
        
        self.projection_selector = pn.widgets.RadioButtonGroup(
            name="Projection Mode", 
            options=["Max Projection", "Select Slice"],
            value="Max Projection",
            width=200
        )
        
        self.z_slice_slider = pn.widgets.IntSlider(
            name="XY View (Z slice)",
            start=0, end=4, value=2,
            width=160,
            disabled=True  # Start disabled
        )
        
        self.y_slice_slider = pn.widgets.IntSlider(
            name="XZ View (Y slice)", 
            start=0, end=19, value=10,
            width=160,
            disabled=True  # Start disabled
        )
        
        # Status text
        self.status_text = pn.pane.Markdown("**Status:** No cluster selected", width=200)
        
        # Display plots (3 channels × 2 views each)
        self.xy_plots = [self._create_empty_plot(f"XY View (20x20)", small=False) for i in range(3)]
        self.xz_plots = [self._create_empty_plot(f"XZ View (5x20)", small=True) for i in range(3)]

        # Set up callbacks
        self.view_selector.param.watch(self._on_view_change, 'value')
        self.projection_selector.param.watch(self._on_projection_change, 'value')
        self.z_slice_slider.param.watch(self._on_z_slice_change, 'value')
        self.y_slice_slider.param.watch(self._on_y_slice_change, 'value')
        self.on_sample_changed = None

        if self.dataset is not None:
            if self.sample_indices is not None and len(self.sample_indices) > 0:
                random_sample_idx = np.random.choice(len(self.sample_indices))
                random_orig_idx = self.sample_indices[random_sample_idx]
                status_msg = f"Showing random sample {random_orig_idx}"
                self.display_dataset = self.dataset[np.sort(self.sample_indices)]
            else:
                n_samples = self.dataset.shape[0]
                random_orig_idx = np.random.choice(n_samples)
                status_msg = f"Showing random sample {random_orig_idx}"
                self.display_dataset = self.dataset
        
            self.current_sample = [random_orig_idx]
            self._update_plots()
            self.status_text.object = status_msg
    
    def _create_empty_plot(self, title, small=False):
        """Create empty bokeh plot for displaying images"""
        height = 70 if small else 140
        y_range = (0, 5) if small else (0, 20)
        plot = figure(
            width=160, height=height,
            title=title,
            toolbar_location=None,
            x_range=(0, 20), y_range=y_range
        )
        plot.axis.visible = False
        plot.grid.visible = False
        plot.title.text_font_size = "10pt"
        return plot
    
    def load_cluster_data(self, cluster_id, display_data, tapped_idx):
        """Load and cache cluster statistics from HDF5"""
        if self.dataset is None:
            self.status_text.object = "**Error:** No HDF5 dataset available"
            return
            
        # Check if already cached and clustering hasn't changed
        if cluster_id in self.cluster_cache:
            self.current_cluster_id = cluster_id
            self.cluster_cache[cluster_id]["selected"] = self.display_dataset[tapped_idx]
            if self.sample_indices is not None:
                original_idx = self.sample_indices[tapped_idx]
                self.cluster_cache[cluster_id]["selected"] = [original_idx]
            else:
                self.cluster_cache[cluster_id]["selected"] = [tapped_idx]
            self._update_current_sample()
            self.status_text.object = f"**Status:** Loaded cached cluster {cluster_id}"
            return
        
        try:
            # Get indices of points in this cluster
            cluster_mask = display_data['cluster'] == cluster_id
            cluster_data = display_data[cluster_mask]
            cluster_original_indices = cluster_data['original_index'].values
            
            if len(cluster_original_indices) == 0:
                self.status_text.object = f"**Error:** No points in cluster {cluster_id}"
                return
            
            sampling_msg = " (sampled)" if self.use_sampling else ""
            self.status_text.object = f"**Status:** Loading {len(cluster_original_indices)} samples{sampling_msg}..."

            sort_order = np.argsort(cluster_original_indices)
            sorted_indices = cluster_original_indices[sort_order]
            cluster_samples_sorted = self.dataset[sorted_indices]
            
            # Load cluster data from HDF5
            unsort_order = np.argsort(sort_order)
            cluster_samples = cluster_samples_sorted[unsort_order]
            
            # Verify expected shape
            if cluster_samples.ndim != 5 or cluster_samples.shape[1:] != (3, 5, 20, 20):
                self.status_text.object = f"**Error:** Unexpected data shape: {cluster_samples.shape}"
                return
            
            # Compute statistics
            cluster_umap = cluster_data[['umap_x', 'umap_y']].values
            com = np.mean(cluster_umap, axis=0)                 # center of mass of UMAP cluster
            distances = np.linalg.norm(cluster_umap - com, axis=1)
            closest_idx = np.argmin(distances)
            median_idx = cluster_original_indices[closest_idx]
            # median_sample = cluster_samples[closest_idx]        # (3, 5, 20, 20)
            # random_idx = np.random.choice(len(cluster_original_indices), size=min(10, len(cluster_original_indices)), replace=False)
            # random_sample = cluster_samples[random_idx]  # (3, 5, 20, 20)
            if self.sample_indices is not None:
                selected_idx = self.sample_indices[tapped_idx]
            else:
                selected_idx = tapped_idx
            
            # Cache results
            self.cluster_cache[cluster_id] = {
                "selected": [selected_idx],
                "mean": sorted_indices,
                "median": [median_idx], 
            }
            
            self.current_cluster_id = cluster_id
            self._update_current_sample()
            
            self.status_text.object = f"**Status:** Loaded cluster {cluster_id} ({len(cluster_original_indices)} samples)"

            if self.on_sample_changed and self.current_sample is not None:
                self.on_sample_changed(self.current_sample)
            
        except Exception as e:
            self.status_text.object = f"**Error:** Failed to load cluster data: {str(e)}"
    
    def _update_current_sample(self):
        """Update current sample based on view selector"""
        if self.current_cluster_id is None or self.current_cluster_id not in self.cluster_cache:
            return
            
        view_type = self.view_selector.value.lower()
        self.current_sample = self.cluster_cache[self.current_cluster_id][view_type]
        self._update_plots()

        if self.on_sample_changed:
            self.on_sample_changed(self.current_sample)
    
    def _on_view_change(self, event):
        """Handle view selector changes"""
        self._update_current_sample()
    
    def _on_projection_change(self, event):
        """Handle projection mode changes"""
        # Update slider enabled/disabled state
        use_slices = (event.new == "Select Slice")
        self.z_slice_slider.disabled = not use_slices
        self.y_slice_slider.disabled = not use_slices
        self._update_plots()
    
    def _on_z_slice_change(self, event):
        """Handle Z slice slider changes"""
        if self.projection_selector.value == "Select Slice":
            self._update_plots(which="top")

    def _on_y_slice_change(self, event):
        """Handle Y slice slider changes"""
        if self.projection_selector.value == "Select Slice":
            self._update_plots(which="bottom")
    
    def _update_plots(self, which="all"):
        """Update all 6 plots based on current sample and settings"""
        if self.current_sample is None:
            return
        if len(self.current_sample) > 1:
            sample_volume = np.mean(self.dataset[self.current_sample], axis=0)
        else:
            sample_volume = self.dataset[self.current_sample[0]]
        
        projection_mode = self.projection_selector.value
        
        for channel in range(3):
            channel_volume = sample_volume[channel]  # (5, 20, 20) - Z, Y, X
            
            # Generate XY and XZ views
            if projection_mode == "Max Projection":
                xy_data = np.max(channel_volume, axis=0)  # Max over Z → (20, 20) - Y, X
                xz_data = np.max(channel_volume, axis=1)  # Max over Y → (5, 20) - Z, X
            else:  # Select Slice
                z_slice = self.z_slice_slider.value
                y_slice = self.y_slice_slider.value
                xy_data = channel_volume[z_slice, :, :]  # (20, 20) - Y, X
                xz_data = channel_volume[:, y_slice, :]  # (5, 20) - Z, X
            
            # Update plots with proper aspect ratios
            if which == "all":
                self._update_single_plot(self.xy_plots[channel], xy_data, (20, 20), channel=channel)
                self._update_single_plot(self.xz_plots[channel], xz_data, (20, 5), channel=channel)
            elif which == "top":
                self._update_single_plot(self.xy_plots[channel], xy_data, (20, 20), channel=channel)
            elif which == "bottom":
                self._update_single_plot(self.xz_plots[channel], xz_data, (20, 5), channel=channel)
    
    def _update_single_plot(self, plot, data, expected_shape, channel):
        """Update a single bokeh plot with 2D data"""
        palettes = [Greys256, Greys256, Blues8[::-1]]
        
        # Update plot ranges to match data
        plot.x_range.end = expected_shape[0]
        plot.y_range.end = expected_shape[1]
        
        # Normalize data for display (0-1 range)
        if data.max() > data.min():
            data_norm = (data - data.min()) / (data.max() - data.min())
        else:
            data_norm = data
        
        # Flip data vertically for proper image orientation (bokeh displays images upside down)
        data_flipped = np.flipud(data_norm)
        
        # Determine which renderer list to use
        if plot in self.xy_plots:
            renderer_list = self.xy_image_renderers
            plot_index = self.xy_plots.index(plot)
        else:
            renderer_list = self.xz_image_renderers
            plot_index = self.xz_plots.index(plot)
        
        # If renderer doesn't exist, create it
        if renderer_list[plot_index] is None:
            renderer_list[plot_index] = plot.image(
                image=[data_flipped], 
                x=0, y=0, 
                dw=expected_shape[0], dh=expected_shape[1],
                palette=palettes[channel]
            )
        else:
            # Update existing renderer's data source
            renderer_list[plot_index].data_source.data = {
                'image': [data_flipped],
                'x': [0],
                'y': [0],
                'dw': [expected_shape[0]],
                'dh': [expected_shape[1]]
            }
    
    def clear_cache(self):
        """Clear cluster cache (call when clustering changes)"""
        self.cluster_cache = {}
        self.current_cluster_id = None
        self.current_sample = None
        self.status_text.object = "**Status:** Cache cleared - select a cluster"
        
        # Clear all plots and reset renderer tracking
        for plot in self.xy_plots + self.xz_plots:
            plot.renderers = []
        
        # Reset renderer tracking
        self.xy_image_renderers = [None, None, None]
        self.xz_image_renderers = [None, None, None]
    
    def get_layout(self):
        """Return the complete BoxViewer layout"""
        # Main controls (without sliders)
        main_controls = pn.Column(
            "### Sample Viewer",
            self.view_selector,
            pn.Spacer(height=10),
            self.projection_selector,
            pn.Spacer(height=20),
            self.status_text,
            width=240,
            margin=(10, 10)
        )
        
        # Create channel columns for plots
        channel_columns = []
        for channel in range(3):
            if channel == 0:
                name = "Image"
            elif channel == 1:
                name = "Correlation Map"
            else:
                name = "Footprint"
            
            channel_header = pn.pane.Markdown(f"**{name}**", 
                                        margin=(5, 0, 0, 0), 
                                        align='center')
            channel_col = pn.Column(
                channel_header,
                pn.pane.Bokeh(self.xy_plots[channel], sizing_mode='fixed'),
                pn.pane.Bokeh(self.xz_plots[channel], sizing_mode='fixed'),
                margin=(5, 5)
            )
            channel_columns.append(channel_col)
        
        # Create slider controls positioned next to the rows they control
        z_slider_control = pn.Column(
            pn.Spacer(height=75),  # Align with XY plots (accounting for header)
            self.z_slice_slider,
            pn.Spacer(height=70),  # Space to align with XZ row
            margin=(5, 10),
            width=180
        )
        
        y_slider_control = pn.Column(
            self.y_slice_slider,
            margin=(5, 10),
            width=180
        )
        
        # Combine sliders into one column
        slider_controls = pn.Column(
            z_slider_control,
            y_slider_control,
            width=180
        )
        
        # Create the plots section: 3 channel columns + slider controls
        plots_section = pn.Row(
            *channel_columns,
            slider_controls,
            margin=(10, 0)
        )
        
        return pn.Column(
            main_controls,
            plots_section,
            width=800
        )
    
    def __del__(self):
        """Clean up HDF5 file handle"""
        if self.hdf5_file is not None:
            try:
                self.hdf5_file.close()
            except:
                pass


