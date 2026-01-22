import os, sys
import numpy as np
import pandas as pd
import panel as pn
from sklearn.cluster import KMeans
import json
from collections import OrderedDict
sys.path.insert(0, 'curation')
from umap_visualiser import UMAPVisualiser
from box_viewer import BoxViewer
from hist_viewer import HistViewer
from sklearn.linear_model import SGDClassifier
import h5py
from umap import UMAP
import time

pn.extension()

# Mapping from HistViewer property names to UMAPVisualiser view modes
PROPERTY_TO_VIEW_MODE = {
    'Footprint Size': 'size',
    'ROI Shot Noise': 'snr',
    'Session': 'session',
    'Edge Cells': 'edge',
    'Mean Intensity': 'intensity',
    'Mean Correlation': 'correlation',
    'Probability': 'prob',
    'Peak Value': 'peak',
    'Voxel SNR': 'vox',
    'Contamination Factor': 'contamination'
}


class AppOrchestrator:
    def __init__(self, umap_file_path, nn_features_path, hdf5_path="data.h5"):
        self.umap_file_path = umap_file_path
        self.hdf5_path = hdf5_path
        self.classifications_file = None
        self.sample_size = 50000
        self.use_sampling = True
        self.nn_features_path = nn_features_path
        
        # Shared data state
        self.umap_embedding = None
        self.nn_features = None
        self.full_data = None
        self.properties = {
            'Footprint Size': None,
            'Mean Intensity': None,
            'Mean Correlation': None,
            'ROI Shot Noise': None,
            'Edge Cells': None,
            'Session': None,
            'Voxel SNR': None,
            'Probability': None,
            'Peak Value': None,
            'Contamination Factor': None
        }
        self.UMAPtime = None
        self.display_data = None
        self.classifications = None
        self.sample_indices = None
        self.cell_probs = None
        self.labelled_features_idx = []
        self.labels = []
        self.available_features = OrderedDict()
        self.curation_features_to_use = ['PCA']
        
        # Components
        self.umap_visualiser = None
        self.box_viewer = None
        self.hist_viewer = None
        self.linear_model = SGDClassifier(loss='log_loss')
        
        # Shared widgets
        self.cluster_slider = pn.widgets.IntSlider(
            name='Number of Clusters', 
            start=3, end=100, value=20, 
            width=200
        )
        
        self.cluster_button = pn.widgets.Button(
            name='Update Clustering',
            button_type='default',
            width=200
        )
        
        # Classification buttons
        self.classify_cluster_cell_button = pn.widgets.Button(
            name='CELL', 
            button_type='danger',
            width=95
        )
        
        self.classify_cluster_not_cell_button = pn.widgets.Button(
            name='NOT CELL', 
            button_type='danger',
            button_style='outline',
            width=95
        )
        
        self.reset_button = pn.widgets.Button(
            name='Reset Classifications', 
            button_type='warning',
            width=200
        )
        
        self.save_button = pn.widgets.Button(
            name='Save Classifications',
            button_type='success',
            width=200
        )

        self.view_toggle = pn.widgets.ToggleGroup(
            name='UMAP view mode',
            options={'Cluster': 'cluster', 'Property': 'property'},
            value='cluster',
            behavior='radio',
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

        # Set up view toggle callback
        self.view_toggle.param.watch(self._on_view_toggle_change, 'value')

        self.UMAPbutton = pn.widgets.Button(
            name=f'Rerun UMAP ({self.UMAPtime}s)' if self.UMAPtime is not None else 'Rerun UMAP',
            button_type='primary',
            width=200
        )
        self.UMAPbutton.on_click(self.recompute_umap)
    
    def load_data(self):
        """Load and prepare all data with sampling coordination"""
        print("Loading data...")
        try:
            if not os.path.exists(self.umap_file_path):
                self.status_text.object = f"**Error:** File not found: {self.umap_file_path}"
                return
            
            # Load UMAP embeddings
            self.umap_embedding = np.load(self.umap_file_path)
            curation_dir = os.path.dirname(self.umap_file_path)
            
            if self.umap_embedding.ndim != 2 or self.umap_embedding.shape[1] != 2:
                self.status_text.object = "**Error:** UMAP file must be 2D with shape (n_points, 2)"
                return
            
            n_points = self.umap_embedding.shape[0]

            # Load NN features
            self.nn_features = np.load(self.nn_features_path)
            if self.nn_features.shape[0] != n_points:
                self.status_text.object = "**Error:** NN features size mismatch with UMAP points"
                print("NN features shape:", self.nn_features.shape)
                print("UMAP embedding shape:", self.umap_embedding.shape)
                return
            
            pc1_var = np.var(self.nn_features[:, 0])
            
            # Check for extra features
            # curation_features.npy is a dict of additional features (no need to include the PCA in this as this is loaded by default anyway)
            # each key should be the name of the feature, and the value should be a numpy array of shape (n_ROIs, feature_dim)
            if os.path.exists(os.path.join(curation_dir, 'curation_features.npy')):
                self.feature_dict = np.load(os.path.join(curation_dir, 'curation_features.npy'), allow_pickle=True).item()
                for key, value in self.feature_dict.items():
                    if value.shape[0] == n_points:
                        if len(np.unique(value)) <= 1:
                            print(f"Ignoring {key} as it has 0 variance")
                            continue
                        print(f"Found property: {key}")
                        value_scaled = value - np.mean(value, axis=0, keepdims=True)
                        value_scaled = value_scaled / (np.std(value_scaled, axis=0, keepdims=True) + 1e-6) * np.sqrt(pc1_var)
                        self.available_features[key] = np.expand_dims(value_scaled, axis=-1) if value_scaled.ndim == 1 else value_scaled

                        # Load the curation features into self.properties for use in UMAP visualiser and hist viewer
                        self.properties[key] = value
                    else:
                        print(f"Ignoring {key} due to size mismatch: {value.shape} vs {n_points} ROIs")
            if 'PCA' not in self.available_features:
                self.available_features['PCA'] = self.nn_features
            self.available_features.move_to_end('PCA', last=False)
            self.features_toggle = pn.widgets.MultiChoice(
                options=list(self.available_features.keys()),
                value=['PCA'],
                solid=False,
                width=200,
            )
            self.features_toggle.param.watch(self.update_curation_features, 'value')

            # Set up classifications file
            self.classifications_file = os.path.join(curation_dir, f"roi_classifications.json")
            
            # Load existing classifications
            self.load_existing_classifications(n_points)
            
            # Determine sampling strategy
            if n_points > self.sample_size:
                self.use_sampling = True
                self.sample_indices = np.random.choice(n_points, self.sample_size, replace=False)
                self.sample_indices.sort()
                self.status_text.object = f"**Status:** Large dataset ({n_points:,} points) - sampling enabled"
            else:
                self.use_sampling = False
                self.sample_indices = np.arange(n_points)
            
            # Initial clustering
            kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
            initial_clusters = kmeans.fit_predict(self.nn_features)

            info_file = os.path.join(curation_dir, 'all_sessions_info.npy')
            if os.path.exists(info_file):
                all_sessions_info = np.load(info_file, allow_pickle=True).item()
                self.UMAPtime = all_sessions_info.get('UMAPtime', None)
            
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
            raise ValueError(f"Could not load data: {str(e)}")
    
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
        self._open_hdf5()

        if self.display_data is not None:

            self.umap_visualiser = UMAPVisualiser(self.display_data, properties=self.properties, sample_indices=self.sample_indices, use_sampling=self.use_sampling)
            # Subscribe to selection events
            self.umap_visualiser.on_cluster_selected = self.on_cluster_selected

            # For box and hist viewers, we need the full dataset (self.dataset), so we pass this in along with the sampling info
            self.box_viewer = BoxViewer(self.dataset, self.sample_indices, self.use_sampling)
            self.hist_viewer = HistViewer(self.dataset, self.properties, self.sample_indices, self.use_sampling)

            if self.box_viewer and self.hist_viewer:
                self.box_viewer.on_sample_changed = self.hist_viewer.update_individual_sample

            # Connect threshold classification callback
            if self.hist_viewer:
                self.hist_viewer.on_threshold_classify = self._classify_by_threshold

            # Sync hist viewer property selection to UMAP view mode
            if self.hist_viewer and self.umap_visualiser:
                self.hist_viewer.property_selector.param.watch(self._on_hist_property_change, 'value')

    def _open_hdf5(self):
        """Open HDF5 file for reading (same pattern as BoxViewer)"""
        try:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')
            
            # Find the main dataset (same logic as BoxViewer)
            self.dataset = self.hdf5_file.get('data', None)
            if self.dataset is None:
                self.status_text.object = "**Error:** Couldn't load the dataset from HDF5 file."
            else:
                self.status_text.object = f"**Status:** Using HDF5 dataset"
            
            # Verify expected shape
            if self.dataset.ndim != 5 or self.dataset.shape[1:] != (3, 5, 20, 20):
                self.status_text.object = f"**Error:** Unexpected data shape: {self.dataset.shape}"
                self.dataset = None

            # Load shot noise if available
            if 'shot_noise' in self.hdf5_file:
                shot_noise = self.hdf5_file['shot_noise'][:]
                if shot_noise.shape[0] == self.dataset.shape[0]:
                    self.properties['ROI Shot Noise'] = np.clip(shot_noise, 0, 1)
                else:
                    print(f"Found shot noise but shape mismatch: {shot_noise.shape} vs {self.dataset.shape} (shot noise vs dataset length)")
                    print("Therefore ignoring shot noise")
            else:
                print("No shot noise in hdf5")
            
            # Load edge cells if available
            if 'edge_cells' in self.hdf5_file:
                edge_cells = self.hdf5_file['edge_cells'][:]
                if edge_cells.shape[0] == self.dataset.shape[0]:
                    self.properties['Edge Cells'] = edge_cells
                else:
                    print(f"Found edge cells but shape mismatch: {edge_cells.shape} vs {self.dataset.shape} (edge cells vs dataset length)")
                    print("Therefore ignoring edge cells")
            else:
                print("No edge cells in hdf5")

            # Load session IDs if available
            if 'session_id' in self.hdf5_file:
                session_id = self.hdf5_file['session_id'][:]
                if session_id.shape[0] == self.dataset.shape[0]:
                    self.properties['Session'] = session_id
                else:
                    print(f"Found session IDs but shape mismatch: {session_id.shape} vs {self.dataset.shape} (session ID vs dataset length)")
                    print("Therefore ignoring session IDs")
            else:
                print("No session IDs in hdf5")

        except Exception as e:
            self.status_text.object = f"**Error:** Could not open HDF5 file: {str(e)}"
    
    def on_cluster_selected(self, cluster_id, tapped_idx):
        """Handle cluster selection from UMAP visualiser"""
        # This will be called when a cluster is selected in the UMAP
        # Update status and prepare for classification
        cluster_size = len(self.full_data[self.full_data['cluster'] == cluster_id])
        self.status_text.object = f"**Status:** Selected cluster {cluster_id} ({cluster_size:,} points) - use cluster classification buttons"
        self.selected_cluster = cluster_id

        # Load cluster data in BoxViewer
        if self.box_viewer:
            self.box_viewer.load_cluster_data(cluster_id, self.display_data, tapped_idx)

        # also load in HistViewer
        if self.hist_viewer:
            self.hist_viewer.load_cluster_data(cluster_id, self.display_data)

    def _on_hist_property_change(self, event):
        """Sync UMAP view mode to match hist viewer property selection (only when in property mode)"""
        if self.view_toggle.value != 'property':
            return
        property_name = event.new
        view_mode = PROPERTY_TO_VIEW_MODE.get(property_name, 'clus')
        if self.umap_visualiser:
            self.umap_visualiser.set_view_mode(view_mode)

    def _on_view_toggle_change(self, event):
        """Handle switching between cluster and property coloring modes"""
        if event.new == 'cluster':
            if self.umap_visualiser:
                self.umap_visualiser.set_view_mode('clus')
        else:  # property mode
            # Use whatever property is currently selected in hist viewer
            if self.hist_viewer and self.umap_visualiser:
                property_name = self.hist_viewer.property_selector.value
                view_mode = PROPERTY_TO_VIEW_MODE.get(property_name, 'clus')
                self.umap_visualiser.set_view_mode(view_mode)

    def update_curation_features(self, event=None):
        """Update the features used for curation based on toggle selection"""
        selected_features = self.features_toggle.value
        if not selected_features:
            self.curation_features_to_use = ['PCA']
            self.features_toggle.value = ['PCA']
        else:
            self.curation_features_to_use = list(selected_features)
        
        # Re-run clustering with new features
        self.update_clusters()

    def update_clusters(self, event=None):
        """Update clustering"""
        if self.full_data is None:
            return
        
        n_clusters = self.cluster_slider.value
        self.status_text.object = f"**Status:** Computing {n_clusters} clusters, please wait..."

        self.curation_features = np.hstack([self.available_features[feat] for feat in self.curation_features_to_use])
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            new_clusters = kmeans.fit_predict(self.curation_features)
            
            # Update full dataset
            self.full_data['cluster'] = new_clusters
            
            # Update display data
            self.prepare_display_data()
            
            # Update UMAP visualiser
            if self.umap_visualiser:
                self.umap_visualiser.update_data(self.display_data)
                self.view_toggle.value = 'cluster'  # Reset to cluster view

            # Update BoxViewer
            if self.box_viewer:
                self.box_viewer.clear_cache()
            
            # Update HistViewer
            if self.hist_viewer:
                self.hist_viewer.clear_cache()
            
            self.status_text.object = f"**Status:** Updated to {n_clusters} clusters"
            
        except Exception as e:
            self.status_text.object = f"**Error:** Clustering failed: {str(e)}"

    def recompute_umap(self, event=None):
        """Recompute UMAP embedding"""
        if self.curation_features is None:
            self.status_text.object = "**Error:** No features selected for UMAP recomputation"
            return
        
        try:
            self.status_text.object = "**Status:** Recomputing UMAP, please wait..."
            self.UMAPbutton.name = "Running UMAP..."
            self.UMAPbutton.button_style = 'outline' 
            start = time.time()
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1, metric='euclidean')
            new_embedding = reducer.fit_transform(self.curation_features)
            
            # Update UMAP embedding
            self.umap_embedding = new_embedding
            self.full_data['umap_x'] = new_embedding[:, 0]
            self.full_data['umap_y'] = new_embedding[:, 1]
            
            # Update display data
            self.prepare_display_data()
            
            # Update UMAP visualiser
            if self.umap_visualiser:
                self.umap_visualiser.update_data(self.display_data, False)
                self.view_toggle.value = 'cluster'  # Reset to cluster view
            
            end = time.time()
            self.UMAPtime = np.round(end - start, 0)
            self.UMAPbutton.name = f"Rerun UMAP ({self.UMAPtime}s)"
            self.UMAPbutton.button_style = 'solid'
            self.status_text.object = "**Status:** UMAP recomputation complete"
        
        except Exception as e:
            self.status_text.object = f"**Error:** UMAP recomputation failed: {str(e)}"
    
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
        clus_idx = np.where(cluster_mask)[0]
        cluster_size = cluster_mask.sum()
        
        self.full_data.loc[cluster_mask, 'classification'] = classification_type
        
        # Update display data
        self.prepare_display_data()
        
        # Update UMAP visualiser
        if self.umap_visualiser:
            self.umap_visualiser.update_data(self.display_data, False)
        
        # Update status
        cell_count = sum(1 for c in self.full_data['classification'] if c == 'cell')
        not_cell_count = sum(1 for c in self.full_data['classification'] if c == 'not_cell')

        self.status_text.object = "Classifying cluster..."

        self.labelled_features_idx.extend(clus_idx)
        self.labels.extend(np.array([1 if classification_type == 'cell' else 0] * clus_idx.shape[0]))

        self.curation_features = np.hstack([self.available_features[feat] for feat in self.curation_features_to_use])

        if len(np.unique(self.labels)) > 1:
            # Fit model if we have positive and negative samples
            self.linear_model.fit(self.curation_features[self.labelled_features_idx], self.labels)
            class_idx = np.argwhere(self.linear_model.classes_ == 1)[0][0]
            self.cell_probs = self.linear_model.predict_proba(self.curation_features[self.sample_indices])[:, class_idx]
            self.umap_visualiser.set_probs(self.cell_probs)

            self.hist_viewer.update_property('Probability', self.cell_probs)

        self.status_text.object = f"**Status:** Classified cluster {self.selected_cluster} ({cluster_size:,} points) as '{classification_type}' | Total - Cells: {cell_count:,}, Not cells: {not_cell_count:,}"

        # Reset selection
        self.selected_cluster = None

    def _classify_by_threshold(self, classification_type):
        """Classify points based on histogram threshold selection (global scope)"""
        if self.hist_viewer is None:
            return

        # Get the threshold mask from hist_viewer
        mask = self.hist_viewer.get_threshold_mask()
        if mask is None or not np.any(mask):
            self.status_text.object = "**Status:** No points in threshold range"
            return

        # Map from population sample indices to original full_data indices
        # The mask corresponds to the values in population_cache, which uses sample_indices
        if self.use_sampling and self.sample_indices is not None:
            threshold_indices = self.sample_indices[mask]
        else:
            threshold_indices = np.where(mask)[0]

        count = len(threshold_indices)

        # Apply classification to full_data
        self.full_data.loc[threshold_indices, 'classification'] = classification_type

        # Update display data
        self.prepare_display_data()

        # Update UMAP visualiser
        if self.umap_visualiser:
            self.umap_visualiser.update_data(self.display_data, False)

        # Update ML model training data
        self.labelled_features_idx.extend(threshold_indices)
        label_val = 1 if classification_type == 'cell' else 0
        self.labels.extend([label_val] * len(threshold_indices))

        self.curation_features = np.hstack([self.available_features[feat] for feat in self.curation_features_to_use])

        if len(np.unique(self.labels)) > 1:
            # Fit model if we have positive and negative samples
            self.linear_model.fit(self.curation_features[self.labelled_features_idx], self.labels)
            class_idx = np.argwhere(self.linear_model.classes_ == 1)[0][0]
            self.cell_probs = self.linear_model.predict_proba(self.curation_features[self.sample_indices])[:, class_idx]
            self.umap_visualiser.set_probs(self.cell_probs)
            self.hist_viewer.update_property('Probability', self.cell_probs)

        # Update status
        cell_count = sum(1 for c in self.full_data['classification'] if c == 'cell')
        not_cell_count = sum(1 for c in self.full_data['classification'] if c == 'not_cell')
        self.status_text.object = f"**Status:** Threshold classified {count:,} points as '{classification_type}' | Total - Cells: {cell_count:,}, Not cells: {not_cell_count:,}"

    def reset_classifications(self, event):
        """Reset all classifications"""
        if self.full_data is None:
            return
        
        self.full_data['classification'] = 'unclassified'
        self.prepare_display_data()
        self.linear_model = SGDClassifier(loss='log_loss')
        self.labelled_features_idx = []
        self.labels = []
        self.cell_probs = None

        if self.umap_visualiser:
            self.umap_visualiser.update_data(self.display_data)
            self.view_toggle.value = 'cluster'  # Reset to cluster view

        if self.hist_viewer:
            self.hist_viewer.remove_property('Probability')

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
            return pn.pane.Markdown("Error loading UMAP visualiser.", width=700)
        
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
            pn.pane.Markdown("### Suite3D Data Curation", width=700, margin=(0, 0, 10, 0)),
            stats_display,
            plot_pane,
            width=720
        )

        classification_controls = [
            "### Cluster Classification",
            pn.Row(self.classify_cluster_cell_button, self.classify_cluster_not_cell_button, width=200),
            pn.Spacer(height=10),
            self.reset_button,
            pn.Spacer(height=10),
            self.save_button,
            pn.Spacer(height=10),
            self.status_text
        ]

        controls = pn.Column(
            self.cluster_slider,
            pn.Spacer(height=10),
            self.cluster_button,
            self.features_toggle,
            self.UMAPbutton,
            self.view_toggle,
            *classification_controls,
            width=300,
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


def create_app(curation_dir):
    """Create the application with orchestrator"""
    umap_file = os.path.join(curation_dir, 'umap_2d.npy')
    nn_features_path = os.path.join(curation_dir, 'pca_embeddings.npy')
    hdf5_path = os.path.join(curation_dir, 'dataset.h5')
    orchestrator = AppOrchestrator(umap_file, nn_features_path=nn_features_path, hdf5_path=hdf5_path)
    return orchestrator.get_layout()

if __name__ == "__main__":

    curation_dir = r"C:\Users\suyash\UCL\tinya_data\rois\curation"
    app = create_app(curation_dir)
    app.servable()

    app.show(port=5007)