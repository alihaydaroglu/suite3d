import os
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import pandas as pd
import panel as pn
from sklearn.cluster import KMeans
import json
from pathlib import Path
from umap_visualiser import UMAPVisualiser
from box_viewer import BoxViewer
from hist_viewer import HistViewer

pn.extension()


class AppOrchestrator:
    def __init__(self, umap_file_path="umap_embeddings.npy", hdf5_path="data.h5"):
        self.umap_file_path = umap_file_path
        self.hdf5_path = hdf5_path
        self.classifications_file = None
        self.sample_size = 50000
        self.use_sampling = True
        
        # Shared data state
        self.umap_embedding = None
        self.full_data = None
        self.display_data = None
        self.classifications = None
        self.sample_indices = None
        
        # Components
        self.umap_visualiser = None
        self.box_viewer = None
        self.hist_viewer = None
        
        # Shared widgets
        self.cluster_slider = pn.widgets.IntSlider(
            name='Number of Clusters', 
            start=10, end=100, value=20, 
            width=200
        )
        
        self.cluster_button = pn.widgets.Button(
            name='Update Clustering',
            button_type='default',
            width=200
        )
        
        # Classification buttons
        self.classify_cluster_cell_button = pn.widgets.Button(
            name='Mark Cluster as CELL', 
            button_type='success',
            width=200
        )
        
        self.classify_cluster_not_cell_button = pn.widgets.Button(
            name='Mark Cluster as NOT CELL', 
            button_type='success',
            width=200
        )
        
        self.reset_button = pn.widgets.Button(
            name='Reset Classifications', 
            button_type='default',
            width=200
        )
        
        self.save_button = pn.widgets.Button(
            name='Save Classifications',
            button_type='success', 
            width=200
        )
        
        # Status text
        self.status_text = pn.pane.Markdown(f"**Status:** Loading {umap_file_path}...", width=200)
        
        # Set up callbacks
        self.cluster_button.on_click(self.update_clusters)
        self.classify_cluster_cell_button.on_click(self.classify_cluster_as_cell)
        self.classify_cluster_not_cell_button.on_click(self.classify_cluster_as_not_cell)
        self.reset_button.on_click(self.reset_classifications)
        self.save_button.on_click(self.save_classifications)
        
        # Initialize
        self.load_data()
        self.create_components()
    
    def load_data(self):
        """Load and prepare all data with sampling coordination"""
        print("Loading UMAP data from:", self.umap_file_path)
        try:
            if not os.path.exists(self.umap_file_path):
                self.status_text.object = f"**Error:** File not found: {self.umap_file_path}"
                return
            
            # Load UMAP embeddings
            self.umap_embedding = np.load(self.umap_file_path)
            
            if self.umap_embedding.ndim != 2 or self.umap_embedding.shape[1] != 2:
                self.status_text.object = "**Error:** UMAP file must be 2D with shape (n_points, 2)"
                return
            
            n_points = self.umap_embedding.shape[0]
            
            # Set up classifications file
            file_stem = Path(self.umap_file_path).stem
            self.classifications_file = os.path.join(os.getcwd(), "curation", f"{file_stem}_classifications.json")
            
            # Load existing classifications
            self.load_existing_classifications(n_points)
            
            # Determine sampling strategy
            if n_points > self.sample_size:
                self.use_sampling = True
                self.sample_indices = np.random.choice(n_points, self.sample_size, replace=False)
                self.status_text.object = f"**Status:** Large dataset ({n_points:,} points) - sampling enabled"
            else:
                self.use_sampling = False
                self.sample_indices = np.arange(n_points)
            
            # Initial clustering
            kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
            initial_clusters = kmeans.fit_predict(self.umap_embedding)
            
            # Create full dataset
            self.full_data = pd.DataFrame({
                'umap_x': self.umap_embedding[:, 0],
                'umap_y': self.umap_embedding[:, 1],
                'cluster': initial_clusters,
                'classification': self.classifications,
                'original_index': np.arange(n_points)
            })
            
            # Prepare display data
            self.prepare_display_data()
            
            display_points = len(self.display_data)
            self.status_text.object = f"**Status:** Loaded {n_points:,} points, displaying {display_points:,} with 20 clusters"

        except Exception as e:
            self.status_text.object = f"**Error:** Could not load data: {str(e)}"
    
    def prepare_display_data(self):
        """Prepare sampled data for display"""
        if self.full_data is None:
            return
        
        if self.use_sampling:
            self.display_data = self.full_data.iloc[self.sample_indices].copy().reset_index(drop=True)
        else:
            self.display_data = self.full_data.copy()
    
    def load_existing_classifications(self, n_points):
        """Load existing classifications if available"""
        self.classifications = ['unclassified'] * n_points
        
        if os.path.exists(self.classifications_file):
            try:
                with open(self.classifications_file, 'r') as f:
                    saved_data = json.load(f)
                    if len(saved_data['classifications']) == n_points:
                        self.classifications = saved_data['classifications']
                        self.status_text.object = f"**Status:** Loaded existing classifications"
                    else:
                        self.status_text.object = f"**Warning:** Classification file size mismatch, starting fresh"
            except Exception as e:
                self.status_text.object = f"**Warning:** Could not load classifications: {str(e)}"
    
    def create_components(self):
        """Create and initialize the visualization components"""
        
        
        if self.display_data is not None:
            self.umap_visualiser = UMAPVisualiser(self.display_data)
            # Subscribe to selection events
            self.umap_visualiser.on_cluster_selected = self.on_cluster_selected

            # add box viewer
            self.box_viewer = BoxViewer(self.hdf5_path, self.sample_indices, self.use_sampling)

            # add hist viewer
            self.hist_viewer = HistViewer(self.hdf5_path, self.sample_indices, self.use_sampling)

            if self.box_viewer and self.hist_viewer:
                self.box_viewer.on_sample_changed = self.hist_viewer.update_individual_sample
    
    def on_cluster_selected(self, cluster_id):
        """Handle cluster selection from UMAP visualiser"""
        # This will be called when a cluster is selected in the UMAP
        # Update status and prepare for classification
        cluster_size = len(self.full_data[self.full_data['cluster'] == cluster_id])
        self.status_text.object = f"**Status:** Selected cluster {cluster_id} ({cluster_size:,} points) - use cluster classification buttons"
        self.selected_cluster = cluster_id

        # Load cluster data in BoxViewer
        if self.box_viewer:
            self.box_viewer.load_cluster_data(cluster_id, self.display_data)
        
        # also load in HistViewer
        if self.hist_viewer:
            self.hist_viewer.load_cluster_data(cluster_id, self.display_data)
    
    def update_clusters(self, event=None):
        """Update clustering"""
        if self.full_data is None:
            return
        
        n_clusters = self.cluster_slider.value
        self.status_text.object = f"**Status:** Computing {n_clusters} clusters, please wait..."
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            new_clusters = kmeans.fit_predict(self.umap_embedding)
            
            # Update full dataset
            self.full_data['cluster'] = new_clusters
            
            # Update display data
            self.prepare_display_data()
            
            # Update UMAP visualiser
            if self.umap_visualiser:
                self.umap_visualiser.update_data(self.display_data)

            # Update BoxViewer
            if self.box_viewer:
                self.box_viewer.clear_cache()
            
            # Update HistViewer
            if self.hist_viewer:
                self.hist_viewer.clear_cache()
            
            self.status_text.object = f"**Status:** Updated to {n_clusters} clusters"
            
        except Exception as e:
            self.status_text.object = f"**Error:** Clustering failed: {str(e)}"
    
    def classify_cluster_as_cell(self, event):
        """Classify entire cluster as cell"""
        self._classify_cluster('cell')
    
    def classify_cluster_as_not_cell(self, event):
        """Classify entire cluster as not cell"""
        self._classify_cluster('not_cell')
    
    def _classify_cluster(self, classification_type):
        """Classify entire cluster"""
        if not hasattr(self, 'selected_cluster') or self.selected_cluster is None:
            self.status_text.object = "**Status:** No cluster selected. Click on a point first."
            return
        
        # Classify all points in the cluster
        cluster_mask = self.full_data['cluster'] == self.selected_cluster
        cluster_size = cluster_mask.sum()
        
        self.full_data.loc[cluster_mask, 'classification'] = classification_type
        
        # Update display data
        self.prepare_display_data()
        
        # Update UMAP visualiser
        if self.umap_visualiser:
            self.umap_visualiser.update_data(self.display_data)
        
        # Update status
        cell_count = sum(1 for c in self.full_data['classification'] if c == 'cell')
        not_cell_count = sum(1 for c in self.full_data['classification'] if c == 'not_cell')
        
        self.status_text.object = f"**Status:** Classified cluster {self.selected_cluster} ({cluster_size:,} points) as '{classification_type}' | Total - Cells: {cell_count:,}, Not cells: {not_cell_count:,}"
        
        # Reset selection
        self.selected_cluster = None
    
    def reset_classifications(self, event):
        """Reset all classifications"""
        if self.full_data is None:
            return
        
        self.full_data['classification'] = 'unclassified'
        self.prepare_display_data()
        
        if self.umap_visualiser:
            self.umap_visualiser.update_data(self.display_data)
        
        self.status_text.object = "**Status:** Reset all classifications"
    
    def save_classifications(self, event):
        """Save current classifications to file"""
        if self.full_data is None or self.classifications_file is None:
            return
        
        try:
            full_classifications = self.full_data['classification'].tolist()
            
            cell_count = sum(1 for c in full_classifications if c == 'cell')
            not_cell_count = sum(1 for c in full_classifications if c == 'not_cell')
            unclassified_count = sum(1 for c in full_classifications if c == 'unclassified')
            
            classifications_data = {
                'filename': self.umap_file_path,
                'n_points': len(self.full_data),
                'classifications': full_classifications,
                'counts': {
                    'cell': cell_count,
                    'not_cell': not_cell_count, 
                    'unclassified': unclassified_count
                }
            }
            
            with open(self.classifications_file, 'w') as f:
                json.dump(classifications_data, f, indent=2)
            
            self.status_text.object = f"**Status:** Saved - Cells: {cell_count:,}, Not cells: {not_cell_count:,}, Unclassified: {unclassified_count:,}"
            
        except Exception as e:
            self.status_text.object = f"**Error:** Could not save classifications: {str(e)}"
    
    def get_layout(self):
        """Return the complete application layout"""
        if self.umap_visualiser is None:
            return pn.pane.Markdown("Loading...")
        
        # Get UMAP plot
        plot_pane = self.umap_visualiser.get_plot_pane()
        
        # Create stats display
        stats_display = pn.pane.Markdown("", width=700, margin=(0, 0, 10, 0))
        
        if self.full_data is not None:
            total_points = len(self.full_data)
            display_points = len(self.display_data) if self.display_data is not None else 0
            
            if self.use_sampling:
                stats_display.object = f"**Total:** {total_points:,} points | **Displaying:** {display_points:,} (sampled)"
            else:
                stats_display.object = f"**{total_points:,} points**"
        
        plot_column = pn.Column(
            pn.pane.Markdown("### Data Curation UMAP", width=700, margin=(0, 0, 10, 0)),
            stats_display,
            plot_pane,
            width=720
        )
        
        controls = pn.Column(
            self.cluster_slider,
            pn.Spacer(height=10),
            self.cluster_button,
            pn.pane.Markdown("*Adjust slider then click 'Update Clustering'*", width=200),
            pn.Spacer(height=20),
            "## Cluster Classification",
            self.classify_cluster_cell_button,
            pn.Spacer(height=10),
            self.classify_cluster_not_cell_button,
            pn.Spacer(height=20),
            "## General Controls",
            self.reset_button,
            pn.Spacer(height=10),
            self.save_button,
            pn.Spacer(height=20),
            self.status_text,
            width=250,
            margin=(10, 10)
        )
        
        top_row_components = [pn.Spacer(width=20), plot_column, pn.Spacer(width=20), controls]
        if self.box_viewer:
            top_row_components.extend([pn.Spacer(width=20), self.box_viewer.get_layout()])

        top_row = pn.Row(
            *top_row_components,
            sizing_mode='stretch_width'
        )

        layout_components = [top_row]

        if self.hist_viewer:
            layout_components.extend([
                pn.Spacer(height=20),
                self.hist_viewer.get_layout()
            ])
    
        return pn.Column(
            *layout_components,
            sizing_mode='stretch_width'
        )


def create_app(umap_file="umap_2d.npy", hdf5_path="data.h5"):
    """Create the application with orchestrator"""
    orchestrator = AppOrchestrator(os.path.join(os.getcwd(), "curation", umap_file), hdf5_path=hdf5_path)
    return orchestrator.get_layout()

app = create_app("contrastive_umap.npy", r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\data\dataset.h5")
app.servable()

if __name__ == "__main__":
    app.show(port=5007)