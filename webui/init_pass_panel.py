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

        self.layout[:3, 0] = self.widgets['reference_volume'].widget
        self.layout[0,  1] = self.widgets['plane_shifts']
        self.layout[1,  1] = self.widgets['registration']
        self.layout[2,  1] = self.widgets['phase_corrs']

        self.summary = None
        self.nz, self.ny, self.nx = (10, 100, 100)

        self.plot_params = {
            'line_width' : 3,
            'colors' : ["#f38b0c","#d34ba1","#0fa4b8","#213073",'#8dda53']
        }


    def load_job(self, job_interface):
        self.summary = job_interface.job_data['summary']
        self.job = job_interface.job

        self.update_all_widgets()

    def update_all_widgets(self):
        #TODO clear plots when loading new job
        self.update_reference_volume()
        self.update_plane_shifts()
        self.update_registration()
        self.update_phase_corrs()


    def update_reference_volume(self):
        reference_volume = self.summary['ref_img_3d']
        self.nz,self.ny,self.nx = reference_volume.shape
        self.widgets['reference_volume'].update_volume(reference_volume)

    def update_plane_shifts(self):
        fig = self.widgets['plane_shifts']
        plane_shifts = self.summary['plane_shifts']

        fig.line(n.arange(self.nz), plane_shifts[:,0], 
                 line_width = self.plot_params['line_width'],
                 color = self.plot_params['colors'][0],
                 legend_label='y', name = 'y')
        fig.line(n.arange(self.nz), plane_shifts[:,1], 
                 line_width = self.plot_params['line_width'],
                 color = self.plot_params['colors'][1],
                 legend_label='x', name = 'x')
        
    
    def update_registration(self):
        fig = self.widgets['registration']
        voxel_size = self.job.params['voxel_size_um']
        fs = self.job.params['fs']
        subpixel_shifts = self.summary['reference_info']['subpix_shifts'][-1]
        nt = subpixel_shifts.shape[0]
        ts = n.arange(nt) / fs

        fig.line(ts, subpixel_shifts[:, 1] * voxel_size[1],
                 line_width = self.plot_params['line_width'],
                 color = self.plot_params['colors'][0],
                 legend_label='y', name = 'y')
        fig.line(ts, subpixel_shifts[:, 2] * voxel_size[2],
                 line_width = self.plot_params['line_width'],
                 color = self.plot_params['colors'][1],
                 legend_label='x', name = 'x')
        fig.line(ts, subpixel_shifts[:, 0] * voxel_size[0],
                 line_width = self.plot_params['line_width'],
                 color = self.plot_params['colors'][2],
                 legend_label='z', name = 'z')


    def update_phase_corrs(self):  
        fig = self.widgets['phase_corrs']
        cmax = self.summary['reference_info']['cmax']  
        frame_use_order = self.summary['reference_info']['frame_use_order']

        n_iter, n_frames = cmax.shape
        fig.line(n.arange(n_iter), cmax[:,(frame_use_order >= n_iter-1)].mean(axis=1),
                 line_width = self.plot_params['line_width'],
                 color = self.plot_params['colors'][3],
                 legend_label='phase_corr', name = 'phase_corr')
