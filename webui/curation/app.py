import os
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import pandas as pd
import panel as pn
from sklearn.cluster import KMeans
import json
from pathlib import Path
from .umap_visualiser import UMAPVisualiser
from .box_viewer import BoxViewer
from .hist_viewer import HistViewer

pn.extension()


class AppOrchestrator:
    def __init__(self, umap_file_path="umap_embeddings.npy", hdf5_path="data.h5"):
        self.umap_file_path = umap_file_path
        self.hdf5_path = hdf5_path
        self.classifications_file = None
        self.sample_size = 50000
        self.use_sampling = True

        # Discover available UMAP files in the same directory
        self._refresh_umap_options()

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

        # Embedding selector
        self.embedding_selector = pn.widgets.Select(
            name="Embedding",
            options=list(self.umap_options.keys()),
            value=Path(umap_file_path).stem if Path(umap_file_path).stem in self.umap_options else list(self.umap_options.keys())[0],
            width=200,
        )
        self.embedding_selector.param.watch(self._on_embedding_change, "value")

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

    def _refresh_umap_options(self):
        """Scan for available umap_2d*.npy files."""
        umap_dir = os.path.dirname(self.umap_file_path)
        available = sorted(Path(umap_dir).glob("umap_2d*.npy"))
        self.umap_options = {p.stem: str(p) for p in available}
        if not self.umap_options:
            stem = Path(self.umap_file_path).stem
            self.umap_options = {stem: self.umap_file_path}

    def load_data(self):
        """Load and prepare all data with sampling coordination"""
        print("Loading UMAP data from:", self.umap_file_path)
        try:
            print("Checking if file exists...")
            if not os.path.exists(self.umap_file_path):
                print("File not found:", self.umap_file_path)
                self.status_text.object = f"**Error:** File not found: {self.umap_file_path}"
                return
            print("File exists, loading...")
            # Load UMAP embeddings
            self.umap_embedding = np.load(self.umap_file_path)
            print("UMAP data shape:", self.umap_embedding.shape)
            if self.umap_embedding.ndim != 2 or self.umap_embedding.shape[1] != 2:
                self.status_text.object = "**Error:** UMAP file must be 2D with shape (n_points, 2)"
                return

            n_points = self.umap_embedding.shape[0]

            # Set up classifications file
            file_stem = Path(self.umap_file_path).stem
            classifications_dir = os.path.dirname(self.umap_file_path)
            self.classifications_file = os.path.join(classifications_dir, f"{file_stem}_classifications.json")
            print("Classifications file:", self.classifications_file)
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
            print("Using sampling:", self.use_sampling)
            # Initial clustering
            kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
            initial_clusters = kmeans.fit_predict(self.umap_embedding)
            print("Initial clustering done")
            # Create full dataset
            self.full_data = pd.DataFrame({
                'umap_x': self.umap_embedding[:, 0],
                'umap_y': self.umap_embedding[:, 1],
                'cluster': initial_clusters,
                'classification': self.classifications,
                'original_index': np.arange(n_points)
            })
            print("Full data prepared with shape:", self.full_data.shape)
            # Prepare display data
            self.prepare_display_data()
            print("Display data prepared with shape:", self.display_data.shape)
            display_points = len(self.display_data)
            self.status_text.object = f"**Status:** Loaded {n_points:,} points, displaying {display_points:,} with 20 clusters"
            print(self.status_text.object)
        except Exception as e:
            self.status_text.object = f"**Error:** Could not load data: {str(e)}"
            print(self.status_text.object)

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
            print("Creating UMAP visualiser...")
            self.umap_visualiser = UMAPVisualiser(self.display_data)
            # Subscribe to selection events
            self.umap_visualiser.on_cluster_selected = self.on_cluster_selected
            print("UMAP visualiser created.")
            print("Creating Box and Hist viewers...")
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

    def _on_embedding_change(self, event):
        """Reload UMAP data when user switches embedding type."""
        new_path = self.umap_options.get(event.new)
        if new_path and os.path.exists(new_path):
            self.umap_file_path = new_path
            self.status_text.object = f"**Status:** Loading {event.new}..."
            self.load_data()
            if self.umap_visualiser and self.display_data is not None:
                self.umap_visualiser.update_data(self.display_data)
            if self.box_viewer:
                self.box_viewer.clear_cache()
            if self.hist_viewer:
                self.hist_viewer.clear_cache()

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
            "## Embedding",
            self.embedding_selector,
            pn.Spacer(height=10),
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


class CurationPanel:
    """Wrapper that integrates curation with the job-loading system.

    All heavy operations (preprocessing, orchestrator creation) run in a
    background thread so the UI stays responsive.
    """

    def __init__(self):
        self.orchestrator = None
        self._container = pn.Column(
            pn.pane.Markdown("**Curation:** Load a job to begin, or curation data not yet generated."),
            sizing_mode="stretch_width",
        )
        self.layout = self._container
        self._loading = False

    def load_job(self, job_interface):
        """Called by webui.py when a job is loaded. Runs heavy work in a thread."""
        import threading

        if self._loading:
            return
        self._loading = True

        job = job_interface.job
        jobdir = Path(job_interface.job_data["jobdir"])

        # Locate stats.npy — try job.dirs["rois"], then jobdir directly
        stats_path = None
        rois_dir = job.dirs.get("rois") if hasattr(job, "dirs") else None
        if rois_dir and os.path.isfile(os.path.join(rois_dir, "stats.npy")):
            stats_path = os.path.join(rois_dir, "stats.npy")
        elif (jobdir / "stats.npy").exists():
            stats_path = str(jobdir / "stats.npy")

        if stats_path is None:
            self._container[:] = [pn.pane.Markdown(
                "**Curation:** No stats.npy found for this job. "
                "Run segmentation first."
            )]
            self._loading = False
            return

        curation_dir = jobdir / "curation"
        curation_dir.mkdir(exist_ok=True)
        outputs_dir = curation_dir / "outputs"
        outputs_dir.mkdir(exist_ok=True)

        h5_path = curation_dir / "dataset.h5"
        umap_path = outputs_dir / "umap_2d.npy"

        needs_preprocess = not h5_path.exists() or not umap_path.exists()

        if needs_preprocess:
            self._container[:] = [pn.pane.Markdown(
                "**Curation:** Generating patches and embeddings in background...\n\n"
                "You can use other tabs while this runs."
            )]

        def _do_load():
            try:
                if needs_preprocess:
                    self._preprocess_job(stats_path, str(curation_dir), str(outputs_dir), jobdir)

                self.orchestrator = AppOrchestrator(
                    umap_file_path=str(umap_path),
                    hdf5_path=str(h5_path),
                )
                self._container[:] = [self.orchestrator.get_layout()]
            except Exception as e:
                self._container[:] = [pn.pane.Markdown(
                    f"**Curation Error:** {e}"
                )]
                import traceback; traceback.print_exc()
            finally:
                self._loading = False

        thread = threading.Thread(target=_do_load, daemon=True)
        thread.start()

    def _preprocess_job(self, stats_path, curation_dir, outputs_dir, jobdir):
        """Generate HDF5 patches, PCA UMAP, and visual embeddings for a job."""
        import h5py
        from .dataloader import Suite3DProcessor
        from .dimensionality_reduction import run_permod_pca_umap

        h5_path = os.path.join(curation_dir, "dataset.h5")

        # Step 1: extract patches if HDF5 doesn't exist
        if not os.path.isfile(h5_path):
            print("Curation: extracting patches...")
            processor = Suite3DProcessor(
                data_dir=str(jobdir.parent),  # parent dir containing the job folder
                output_dir=curation_dir,
                box_size=(5, 20, 20),
            )
            # Process just this single session
            session_info = processor.process_and_save_session(jobdir)
            if session_info is None:
                raise RuntimeError("No cells found in this job")

            # Build HDF5 from the saved patches
            patches_file = os.path.join(curation_dir, f"{jobdir.name}_patches.npy")
            data = np.load(patches_file)
            with h5py.File(h5_path, "w") as hf:
                hf.create_dataset("data", data=data)
            os.remove(patches_file)
            print(f"Curation: created {h5_path} with {data.shape[0]} cells")

        # Step 2: run PCA + UMAP if not done
        umap_path = os.path.join(outputs_dir, "umap_2d.npy")
        if not os.path.isfile(umap_path):
            print("Curation: running PCA + UMAP...")
            with h5py.File(h5_path, "r") as f:
                X = f["data"][:]
            run_permod_pca_umap(
                X=X,
                ncomp_per_mod=32,
                batch_size=4096,
                out_dir=outputs_dir,
                whiten=True,
                seed=0,
                umap_neighbors=30,
                umap_min_dist=0.1,
                savename="umap_2d",
            )

        # Step 3: run visual embeddings + UMAP if not done
        visual_umap_path = os.path.join(outputs_dir, "umap_2d_visual.npy")
        if not os.path.isfile(visual_umap_path):
            try:
                print("Curation: running visual embeddings...")
                from .visual_embeddings import run_visual_embedding_pipeline
                from .dimensionality_reduction import run_visual_umap, run_combined_umap

                emb_path = run_visual_embedding_pipeline(
                    stats_path=stats_path,
                    out_dir=outputs_dir,
                    backbone_name="dinov2",
                    batch_size=64,
                    box_size=(5, 20, 20),
                )
                run_visual_umap(
                    embeddings_path=emb_path,
                    out_dir=outputs_dir,
                    savename="umap_2d_visual",
                )
                # Also generate combined embedding
                pca_emb = os.path.join(outputs_dir, "pca_embeddings.npy")
                if os.path.isfile(pca_emb):
                    run_combined_umap(
                        pca_embeddings_path=pca_emb,
                        visual_embeddings_path=emb_path,
                        out_dir=outputs_dir,
                        savename="umap_2d_combined",
                    )
            except Exception as e:
                print(f"Curation: visual embeddings failed (non-fatal): {e}")
                # PCA UMAP still works, visual is optional


def get_curation_panel():
    """Get the curation panel for embedding in the main webapp."""
    return CurationPanel()


if __name__ == "__main__":
    # Standalone mode with hardcoded paths for development
    orchestrator = AppOrchestrator(
        umap_file_path=r"/home/ali/packages/s3d-dev/devbooks/webui_data/outputs/umap_2d.npy",
        hdf5_path=r"/home/ali/packages/s3d-dev/devbooks/webui_data/dataset.h5",
    )
    app = orchestrator.get_layout()
    app.servable()
    app.show(port=5007)
