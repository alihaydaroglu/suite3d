import numpy as np
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, TapTool, ColorBar, LinearColorMapper
from bokeh.palettes import Category20, Reds256, Inferno256, Viridis256
from bokeh.colors import RGB

class UMAPVisualiser:
    """Pure UMAP visualisation component - handles only plot rendering and interactions"""

    def __init__(self, data=None, properties=None, sample_indices=None, use_sampling=False):
        """
        Initialize with data DataFrame containing: umap_x, umap_y, cluster, classification

        Args:
            data: DataFrame with umap_x, umap_y, cluster, classification columns
            properties: dict containing ROI properties for coloring:
                - shot_noise: array of shot noise values
                - footprint_size: array of footprint sizes
                - edge_cells: array of edge cell flags
                - session_id: array of session IDs
                - mean_intensity: array of mean intensity values
                - mean_correlation: array of mean correlation values
                - peak_value: array of peak values
                - contamination: array of contamination values
            sample_indices: indices to sample from properties if use_sampling is True
            use_sampling: whether to sample properties using sample_indices
        """
        self.data = data
        self.properties = properties.copy()
        for key, value in properties.items():
            if value is None:
                continue
            else:
                if use_sampling:
                    self.properties[key] = value[sample_indices]
                else:
                    self.properties[key] = value
        
        self.source = None
        self.plot = None
        self.scatter = None
        self.cluster_colors = Category20[20]
        self.view_mode = 'clus'
        self.last_clicked_cluster = None
        self.prob_mapper = LinearColorMapper(palette=Reds256[::-1], low=0, high=1)
        self.color_bar = None
        
        # Callback for cluster selection - set by orchestrator
        self.on_cluster_selected = None
        
        # Create plot
        self.create_empty_plot()
        
        # Initial render if data provided
        if self.data is not None:
            self.update_plot()

    def set_probs(self, probs):
        """Set probabilities for coloring points"""
        if self.data is not None and probs.shape[0] != self.data.shape[0]:
            print(self.data.shape, probs.shape)
            raise ValueError("Length of probabilities must match number of data points.")
        self.properties['Probability'] = probs

        if self.data is not None:
            self.update_plot()
    
    def set_view_mode(self, mode):
        """Set view mode directly with a string value"""
        mode = mode.lower()
        valid_modes = ['clus', 'prob', 'snr', 'size', 'session', 'edge', 'intensity', 'correlation', 'peak', 'contamination', 'vox']
        if mode not in valid_modes:
            raise ValueError(f"View mode must be one of {valid_modes}.")
        self.view_mode = mode
        self.update_plot()

    def update_data(self, new_data, cluster_mode=True):
        """Update with new data and re-render"""
        self.data = new_data
        if cluster_mode:
            self.view_mode = 'clus'  # Reset to cluster mode on data update
        self.update_plot()
    
    def get_plot_pane(self):
        """Return Panel pane containing the plot"""
        return pn.pane.Bokeh(self.plot, sizing_mode='stretch_width')
    
    def create_empty_plot(self):
        """Create empty plot optimized for large datasets"""
        base_tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset,save"
        
        self.plot = figure(
            width=700, height=500,
            title="",
            tools=base_tools,
            toolbar_location="right",
            output_backend="webgl"  # Use WebGL for better performance
        )
        
        # Style the plot
        self.plot.xaxis.axis_label = "UMAP Dimension 1"
        self.plot.yaxis.axis_label = "UMAP Dimension 2"
        
        # Optimize plot settings for performance
        self.plot.toolbar.autohide = True
        
        # Set up initial tools
        self.update_plot_tools()
    
    def update_plot_tools(self):
        """Update plot tools based on current mode"""
        if self.plot is None:
            return
        
        # Clear existing tools and rebuild
        self.plot.toolbar.tools = []
        
        # Add basic navigation tools
        from bokeh.models import PanTool, WheelZoomTool, BoxZoomTool, ResetTool, SaveTool
        self.plot.add_tools(PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), SaveTool())
        
        # Add tap tool for cluster selection
        tap_tool = TapTool()
        self.plot.add_tools(tap_tool)
    
    def on_point_tap(self, attr, old, new):
        """Handle point tap in cluster mode"""
        if not new or self.data is None:
            return
        
        # Get the cluster of the tapped point
        tapped_idx = new[0]
        if tapped_idx < len(self.data):
            self.last_clicked_cluster = self.data.iloc[tapped_idx]['cluster']
            
            # Notify orchestrator if callback is set
            if self.on_cluster_selected:
                self.on_cluster_selected(self.last_clicked_cluster, tapped_idx)
    
    def get_point_colors_and_properties(self):
        """Color assignment using vectorized operations"""
        if self.data is None or len(self.data) == 0:
            return [], []
        
        n_points = len(self.data)
        
        # Pre-allocate arrays
        colors = np.empty(n_points, dtype=object)
        alphas = np.full(n_points, 0.4)
        
        # Get classification and cluster arrays
        classifications = self.data['classification'].values
        clusters = self.data['cluster'].values
        
        # Vectorized color assignment
        cell_mask = classifications == 'cell'
        not_cell_mask = classifications == 'not_cell'
        unclassified_mask = ~(cell_mask | not_cell_mask)
        
        if self.view_mode == "prob" and self.properties.get('Probability', None) is not None:
            # For all ROIs, use probabilities to set colors
            for i in range(n_points):
                prob = self.properties['Probability'][i]
                colors[i] = RGB(r=255, g=int(255 * (1 - prob)), b=int(255 * (1 - prob)))
        elif self.view_mode == "snr":
            # For all ROIs, use shot noise to set colors
            shot = self.properties.get('ROI Shot Noise', None)
            norm_noise = (shot - np.min(shot)) / (np.max(shot) - np.min(shot) + 1e-8)
            for i in range(n_points):
                noise_val = norm_noise[i]
                colors[i] = Inferno256[int(noise_val * 255)]
        elif self.view_mode == "size":
            # For all ROIs, use footprint size to set colors
            size = self.properties.get('Footprint Size', None)
            norm_size = (size - np.min(size)) / (np.max(size) - np.min(size) + 1e-8)
            for i in range(n_points):
                size_val = norm_size[i]
                colors[i] = Viridis256[int(size_val * 255)]
        elif self.view_mode == "session":
            # For all ROIs, use session ID to set colors
            session_id = self.properties.get('Session', None)
            unique_sessions = np.unique(session_id)
            session_color_map = {sess: Category20[20][i % 20] for i, sess in enumerate(unique_sessions)}
            for i in range(n_points):
                sess_id = session_id[i]
                colors[i] = session_color_map[sess_id]
        elif self.view_mode == "edge":
            # For all ROIs, use edge cell status to set colors
            for i in range(n_points):
                if self.properties['Edge Cells'][i]:
                    colors[i] = 'red'
                else:
                    colors[i] = 'blue'
        elif self.view_mode == "intensity":
            # For all ROIs, use mean intensity to set colors
            mean_intensity = self.properties.get('Mean Intensity', None)
            norm_val = (mean_intensity - np.min(mean_intensity)) / (np.max(mean_intensity) - np.min(mean_intensity) + 1e-8)
            for i in range(n_points):
                colors[i] = Viridis256[int(norm_val[i] * 255)]
        elif self.view_mode == "correlation":
            # For all ROIs, use mean correlation to set colors
            corr = self.properties.get('Mean Correlation', None)
            norm_val = (corr - np.min(corr)) / (np.max(corr) - np.min(corr) + 1e-8)
            for i in range(n_points):
                colors[i] = Inferno256[int(norm_val[i] * 255)]
        elif self.view_mode == "peak":
            # For all ROIs, use peak value to set colors
            peak = self.properties.get('Peak Value', None)
            norm_val = (peak - np.min(peak)) / (np.max(peak) - np.min(peak) + 1e-8)
            for i in range(n_points):
                colors[i] = Viridis256[int(norm_val[i] * 255)]
        elif self.view_mode == "contamination":
            # For all ROIs, use contamination to set colors
            contamination = self.properties.get('Contamination Factor', None)
            norm_val = (contamination - np.min(contamination)) / (np.max(contamination) - np.min(contamination) + 1e-8)
            for i in range(n_points):
                colors[i] = Reds256[int(norm_val[i] * 255)]
        elif self.view_mode == "vox":
            # For all ROIs, use voxel SNR to set colors
            vox_snr = self.properties.get('Voxel SNR', None)
            norm_val = (vox_snr - np.min(vox_snr)) / (np.max(vox_snr) - np.min(vox_snr) + 1e-8)
            for i in range(n_points):
                colors[i] = Inferno256[int(norm_val[i] * 255)]
        else:
            # Cluster mode
            colors[cell_mask] = 'black'
            colors[not_cell_mask] = 'gray'
            # For unclassified, use cluster colors
            for i in np.where(unclassified_mask)[0]:
                colors[i] = self.cluster_colors[clusters[i] % len(self.cluster_colors)]
        
        # Set alphas
        alphas[cell_mask] = 0.8
        alphas[not_cell_mask] = 0.05
        
        return colors.tolist(), alphas.tolist()
    
    def update_plot(self):
        """Update the plot with current data"""
        if self.data is None or len(self.data) == 0:
            return
        
        # Get colors efficiently
        colors, alphas = self.get_point_colors_and_properties()
        
        # Prepare data source
        self.source = ColumnDataSource(data=dict(
            x=self.data['umap_x'].values,
            y=self.data['umap_y'].values,
            cluster=self.data['cluster'].values,
            classification=self.data['classification'].values,
            colors=colors,
            alphas=alphas
        ))
        
        # Clear existing renderers
        self.plot.renderers = []
        
        # Add hover tool for smaller datasets
        if len(self.data) < 50000:
            hover = HoverTool(tooltips=[
                ("Cluster", "@cluster"),
                ("Classification", "@classification")
            ])
            # Remove existing hover tools first
            self.plot.toolbar.tools = [tool for tool in self.plot.toolbar.tools if not isinstance(tool, HoverTool)]
            self.plot.add_tools(hover)
        
        # Create scatter plot
        self.scatter = self.plot.scatter(
            'x', 'y', 
            source=self.source,
            size=3,
            color='colors',
            alpha='alphas',
            selection_color="red",
            nonselection_alpha=0.1
        )

        no_color_bar = ['clus', 'session', 'edge']

        if self.view_mode in no_color_bar and self.color_bar is not None and self.color_bar in self.plot.right:
            # if in cluster mode, remove color bar if present
            self.plot.right.remove(self.color_bar)
        elif self.view_mode in no_color_bar:
            self.color_bar = None
        else:
            if self.color_bar is not None and self.color_bar in self.plot.right:
                self.plot.right.remove(self.color_bar)

            if self.view_mode == 'prob':
                self.color_bar = ColorBar(
                    color_mapper=self.prob_mapper,
                    label_standoff=12,
                    border_line_color=None,
                    location=(0, 0),
                    title="Probability"
                )

            elif self.view_mode == 'snr':
                self.color_bar = ColorBar(
                    color_mapper=LinearColorMapper(palette=Inferno256, low=np.min(self.properties['ROI Shot Noise']), high=np.max(self.properties['ROI Shot Noise'])),
                    label_standoff=12,
                    border_line_color=None,
                    location=(0, 0),
                    title="Shot Noise"
                )

            elif self.view_mode == 'size':
                self.color_bar = ColorBar(
                    color_mapper=LinearColorMapper(palette=Viridis256, low=np.min(self.properties['Footprint Size']), high=np.max(self.properties['Footprint Size'])),
                    label_standoff=12,
                    border_line_color=None,
                    location=(0, 0),
                    title="Footprint Size"
                )

            elif self.view_mode == 'intensity':
                self.color_bar = ColorBar(
                    color_mapper=LinearColorMapper(palette=Viridis256, low=np.min(self.properties['Mean Intensity']), high=np.max(self.properties['Mean Intensity'])),
                    label_standoff=12,
                    border_line_color=None,
                    location=(0, 0),
                    title="Mean Intensity"
                )

            elif self.view_mode == 'correlation':
                self.color_bar = ColorBar(
                    color_mapper=LinearColorMapper(palette=Inferno256, low=np.min(self.properties['Mean Correlation']), high=np.max(self.properties['Mean Correlation'])),
                    label_standoff=12,
                    border_line_color=None,
                    location=(0, 0),
                    title="Mean Correlation"
                )

            elif self.view_mode == 'peak':
                self.color_bar = ColorBar(
                    color_mapper=LinearColorMapper(palette=Viridis256, low=np.min(self.properties['Peak Value']), high=np.max(self.properties['Peak Value'])),
                    label_standoff=12,
                    border_line_color=None,
                    location=(0, 0),
                    title="Peak Value"
                )

            elif self.view_mode == 'contamination':
                self.color_bar = ColorBar(
                    color_mapper=LinearColorMapper(palette=Reds256, low=np.min(self.properties['Contamination Factor']), high=np.max(self.properties['Contamination Factor'])),
                    label_standoff=12,
                    border_line_color=None,
                    location=(0, 0),
                    title="Contamination"
                )

            elif self.view_mode == 'vox':
                self.color_bar = ColorBar(
                    color_mapper=LinearColorMapper(palette=Inferno256, low=np.min(self.properties['Voxel SNR']), high=np.max(self.properties['Voxel SNR'])),
                    label_standoff=12,
                    border_line_color=None,
                    location=(0, 0),
                    title="Voxel SNR"
                )

            self.plot.add_layout(self.color_bar, 'right')
        
        # Set up tap callback for cluster mode
        if self.source:
            self.source.selected.on_change('indices', self.on_point_tap)
        
    def get_selected_indices(self):
        """Get currently selected point indices"""
        if self.source is None:
            return []
        return self.source.selected.indices
    
    def clear_selection(self):
        """Clear current selection"""
        if self.source is not None:
            self.source.selected.indices = []