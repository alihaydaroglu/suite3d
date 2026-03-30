import numpy as n
import os
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

# Create curation tab
curation_layout = get_curation_panel()

ui = pn.Tabs(
    ("Job Interface", job_interface.job_widget),
    ("Initialization", init_panel.layout),
    ("Registration", reg_panel.layout),
    ("Correlation Map", corrmap_panel.layout),
    ("Footprints", footprint_panel.layout),
    ("Extraction Sweeps", sweep_panel.layout),
    ("Curation", curation_layout),
)

def job_load_callback(value):
    if value:
        try:
            init_panel.load_job(job_interface)
        except Exception as e:
            print("Could not load init panel:", e)
        try:
            reg_panel.load_job(job_interface)
        except Exception as e:
            print("Could not load registration panel:", e)
        try:
            corrmap_panel.load_job(job_interface)
        except Exception as e:
            print("Could not load corrmap panel:", e)
        try:
            footprint_panel.load_job(job_interface)
        except Exception as e:
            print("Could not load footprint panel:", e)
        try:
            sweep_panel.load_job(job_interface)
        except Exception as e:
            print("Could not load sweep panel:", e)

pn.bind(job_load_callback, job_interface.param.job_loaded, watch=True)
