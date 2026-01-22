import os
os.environ["OMP_NUM_THREADS"] = "6"
import sys
from pathlib import Path

import panel as pn

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, RangeSlider
from bokeh.plotting import figure, show
from bokeh.models import LinearColorMapper
from bokeh.models import FileInput
from bokeh.layouts import gridplot
from bokeh.palettes import Turbo256,Greys256

# this adds the general suite3d directory to the python path
# it's a hack, and it only works if you call this function from the main suite3d dir
# eventually, the package manager should add packages/suite3d (or the general root dir) to PYTHONPATH during a proper installation
sys.path.insert(0,'.')
from webui.volume_vis import VolumeWidget
from webui.job_interface import JobInterface
from webui.init_pass_panel import InitPanel
from webui.corrmap_panel import CorrmapPanel
from webui.registration_panel import RegistrationPanel
from webui.curation_panel import CurationPanel


pn.extension(design="native")

job_interface = JobInterface(width=None, height=None)

job_widget_vis_button = job_interface.job_widget.controls(['visible'])[1]
job_widget_vis_button.name = 'Show create/load job widget'

init_panel = InitPanel(max_height=800)
reg_panel = RegistrationPanel(max_height=800)
corrmap_panel = CorrmapPanel(max_height=800)
curation_panel = CurationPanel(max_height=800)


ui = pn.Tabs(job_interface.job_widget, init_panel.layout, reg_panel.layout, corrmap_panel.layout, curation_panel.layout)

def job_load_callback(value):
    if value:
        try:
            init_panel.load_job(job_interface)
        except:
            print("Could not load init panel")
        try:
            reg_panel.load_job(job_interface)
        except:
            print("Could not load registration panel")
        try:
            corrmap_panel.load_job(job_interface)
        except:
            print("Could not load corrmap panel")
        try:
            curation_panel.load_job(job_interface)
        except:
            print("Could not load curation panel")

pn.bind(job_load_callback, job_interface.param.job_loaded, watch=True)

ui.servable()
