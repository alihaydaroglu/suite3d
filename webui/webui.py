import numpy as n
import os
import threading
from pathlib import Path

import panel as pn

from .volume_vis import VolumeWidget
from .job_interface import JobInterface
from .init_pass_panel import InitPanel
from .corrmap_panel import CorrmapPanel
from .registration_panel import RegistrationPanel
from .footprint_panel import FootprintPanel
from .sweep_panel import SweepPanel
from .curation.app import get_curation_panel


pn.extension(design="native")

job_interface = JobInterface(width=None, height=None)

job_widget_vis_button = job_interface.job_widget.controls(['visible'])[1]
job_widget_vis_button.name = 'Show create/load job widget'

init_panel = InitPanel(max_height=800)
reg_panel = RegistrationPanel(max_height=800)
corrmap_panel = CorrmapPanel(max_height=800)
footprint_panel = FootprintPanel(max_height=800)
sweep_panel = SweepPanel(max_height=800)

# Curation panel created lazily — just a lightweight wrapper until first use
curation_panel = get_curation_panel()

# Track which panels have been loaded for the current job
_panels_loaded = set()
_job_interface_ref = None

# Map tab index -> (panel, name) for lazy loading
_tab_panels = {
    1: (init_panel, "init"),
    2: (reg_panel, "registration"),
    3: (corrmap_panel, "corrmap"),
    4: (footprint_panel, "footprint"),
    5: (sweep_panel, "sweep"),
    6: (curation_panel, "curation"),
}

ui = pn.Tabs(
    ("Job Interface", job_interface.job_widget),
    ("Initialization", init_panel.layout),
    ("Registration", reg_panel.layout),
    ("Correlation Map", corrmap_panel.layout),
    ("Footprints", footprint_panel.layout),
    ("Extraction Sweeps", sweep_panel.layout),
    ("Curation", curation_panel.layout),
)


def _load_panel_for_tab(tab_idx):
    """Load data for a panel if not already loaded for current job."""
    global _job_interface_ref
    if _job_interface_ref is None:
        return
    if tab_idx in _panels_loaded:
        return
    if tab_idx not in _tab_panels:
        return

    panel, name = _tab_panels[tab_idx]
    try:
        print(f"  Lazy-loading {name} panel...")
        panel.load_job(_job_interface_ref)
        _panels_loaded.add(tab_idx)
    except Exception as e:
        print(f"Could not load {name} panel: {e}")


def _on_tab_change(event):
    """Called when user switches tabs — triggers lazy load."""
    _load_panel_for_tab(event.new)


ui.param.watch(_on_tab_change, "active")


def job_load_callback(value):
    """When a job is loaded, only load the currently active tab's panel."""
    global _job_interface_ref, _panels_loaded
    if not value:
        return

    _job_interface_ref = job_interface
    _panels_loaded.clear()

    # Only load the currently visible tab
    active = ui.active
    _load_panel_for_tab(active)


pn.bind(job_load_callback, job_interface.param.job_loaded, watch=True)
