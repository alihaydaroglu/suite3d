import numpy as n 
import os
import sys
from pathlib import Path


import panel as pn
import param
from bokeh.plotting import figure

# this adds the general suite3d directory to the python path
# it's a hack, and it only works if you call this function from the main suite3d dir
# eventually, the package manager should add packages/suite3d (or the general root dir) to PYTHONPATH during a proper installation
sys.path.insert(0,'.')
from webui.volume_vis import VolumeWidget
from webui.job_interface import JobInterface


class InitPanel(param.Parameterized):
    def __init__(self, max_height = 800):
        super().__init__()

        self.layout = pn.GridSpec(name='Initial Pass', sizing_mode = 'scale_both', max_height=max_height)

        self.widgets = {
            'reference_volume' : VolumeWidget('Reference Volume'),
            'plane_shifts'     : figure(width=400, height=400, sizing_mode='scale_both', title='Plane Shifts'),
            'registration'     : figure(width=400, height=400, sizing_mode='scale_both', title='Registration'),
            'phase_corrs'      : figure(width=400, height=400, sizing_mode='scale_both', title='Phase Correlation'),
        }

        self.layout[:3, 0] = self.widgets['reference_volume']
        self.layout[0,  1] = self.widgets['plane_shifts']
        self.layout[1,  1] = self.widgets['registration']
        self.layout[2,  1] = self.widgets['phase_corrs']

        self.summary = None
        # self.nz, self.ny, 



    def update_reference_volume(self):
        reference_volume = self.summary['ref_img_3d']
        self.nz,self.ny,self.nx = reference_volume.shape[0]