import panel as pn
import param
import numpy as np
import h5py
from pathlib import Path
import sys

sys.path.insert(0, '.')
from curation.app import AppOrchestrator


class CurationPanel(param.Parameterized):
    """Panel wrapper for the curation app to integrate with webui"""

    def __init__(self, max_height=800):
        super().__init__()
        self.max_height = max_height
        self.orchestrator = None
        self.job = None

        # Create placeholder layout
        self.layout = pn.Column(
            pn.pane.Markdown("## Curation", width=700),
            pn.pane.Markdown("Load a job to begin curation.", width=700),
            name="Curation",
            max_height=max_height
        )

    def load_job(self, job_interface):
        """Load data from Suite3D job and initialize curation app"""
        try:
            self.job = job_interface.job
            jobdir = Path(job_interface.job_data['jobdir'])

            # Check if curation data exists
            curation_dir = jobdir / 'rois' / 'curation'
            if not curation_dir.exists():
                self.layout[:] = [
                    pn.pane.Markdown("## Curation", width=700),
                    pn.pane.Markdown(
                        "**No curation data found.**\n\n",
                        width=700
                    )
                ]
                return

            # Load curation files
            umap_file = str(curation_dir / 'umap_2d.npy')
            features_file = str(curation_dir / 'pca_embeddings.npy')
            hdf5_file = str(curation_dir / 'dataset.h5')

            # Check if files exist
            if not all(Path(f).exists() for f in [umap_file, features_file, hdf5_file]):
                missing = [f for f in [umap_file, features_file, hdf5_file] if not Path(f).exists()]
                self.layout[:] = [
                    pn.pane.Markdown("## Curation", width=700),
                    pn.pane.Markdown(
                        f"**Missing curation files:**\n\n" + "\n".join(f"- {Path(f).name}" for f in missing),
                        width=700
                    )
                ]
                return

            # Create orchestrator with job data
            self.orchestrator = AppOrchestrator(umap_file, features_file, hdf5_file)

            # Replace placeholder with actual curation interface
            self.layout[:] = [self.orchestrator.get_layout()]

        except Exception as e:
            self.layout[:] = [
                pn.pane.Markdown("## Curation", width=700),
                pn.pane.Markdown(f"**Error loading curation data:**\n\n{str(e)}", width=700)
            ]
            print(f"Error in CurationPanel.load_job: {e}")
