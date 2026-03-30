import numpy as n
import os
from pathlib import Path

import panel as pn
import param
from bokeh.plotting import figure

from .volume_vis import VolumeWidget
from .job_interface import JobInterface

class CorrmapPanel(param.Parameterized):
    def __init__(self, max_height=None):
        super().__init__()

        self.layout = pn.GridSpec(name = 'Correlation Map', sizing_mode = 'scale_both',
                                  max_height = max_height)
        self.widgets = {
            'dir_selection' : pn.widgets.TextInput(name = 'Dir Name', value = ''),
            'corr_volume' : VolumeWidget('Correlation Map')
        }
        pn.bind(self.update_dir, value = self.widgets['dir_selection'], watch=True)

        self.layout[:3, 0] = self.widgets['corr_volume'].widget
        self.layout[ 0, 1] = self.widgets['dir_selection']

        self.dir_key = None

    def update_dir(self, value):
        if len(value) > 1:
            self.dir_key = value
            self.update_volume()


    def load_job(self, job_interface):
        self.summary = job_interface.job_data['summary']
        self.job = job_interface.job

        self.update_all_widgets()

    def update_all_widgets(self):
        self.update_volume()

    def update_volume(self):
        if (self.dir_key is None and 'corrmap' in self.job.dirs.keys() )or self.dir_key in self.job.dirs.keys():
            
            results = self.job.load_corr_map_results(self.dir_key)
            vol = results['vmap']
            self.nz, self.ny, self.nx = vol.shape
            self.widgets['corr_volume'].update_volume(vol)