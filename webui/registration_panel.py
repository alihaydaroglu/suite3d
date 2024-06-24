import numpy as n 
import os
import sys
from pathlib import Path


import panel as pn
import param
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

# this adds the general suite3d directory to the python path
# it's a hack, and it only works if you call this function from the main suite3d dir
# eventually, the package manager should add packages/suite3d (or the general root dir) to PYTHONPATH during a proper installation
sys.path.insert(0,'.')
from webui.volume_vis import VolumeWidget
from webui.job_interface import JobInterface

class RegistrationPanel(param.Parameterized):
    def __init__(self, max_height=None):
        super().__init__()

        self.layout = pn.GridSpec(name = 'Registration', sizing_mode = 'stretch_both',
                                  max_height = max_height)
        self.widgets = {
            'plane_selection' : pn.widgets.IntInput(name = 'Plane', value = 0, start=0, end=100, step=1, max_height=50),
            'mean_img_plane' : VolumeWidget('Registered Mean Image', slider_text='File #'),
            'shifts' : figure(width=400, height=400, sizing_mode='scale_both', title='Computed Shifts (um)'),
            'volume_quality_over_z' : figure(width=400, height=400, sizing_mode='scale_both', title='Volume Quality over depth'),
            'volume_quality_over_time' : figure(width=400, height=400, sizing_mode='scale_both', title='Volume Quality over time')
        }
        pn.bind(self.update_all_widgets, plane_idx = self.widgets['plane_selection'], watch=True)

        self.layout[:3, 0] = self.widgets['mean_img_plane'].widget
        # self.widgets['mean_img_plane'].widget.append(self.widgets['plane_selection'])
        self.layout[ 4, 0] = self.widgets['plane_selection']
        self.layout[ 0, 1] = self.widgets['shifts']
        self.layout[ 1, 1] = self.widgets['volume_quality_over_time']
        self.layout[ 2, 1] = self.widgets['volume_quality_over_z']

        self.dir_key = None

        self.plot_params = {
            'line_width' : 3,
            'colors' : ["#f38b0c","#d34ba1","#0fa4b8","#213073",'#8dda53']
        }

        self.lines = {}



    def load_job(self, job_interface):
        self.summary = job_interface.job_data['summary']
        self.nz, self.ny, self.nx = self.summary['ref_img_3d'].shape
        self.job = job_interface.job

        self.metric_files = self.job.get_registered_files(filename_filter='reg_metrics')
        self.mean_image_files = self.job.get_registered_files(filename_filter='mean')
        print("Loading images")
        self.imgs = [n.load(img,mmap_mode = 'r') for img in self.mean_image_files]
        print("Loaded images")

        quality_dict = n.load(self.metric_files[0],allow_pickle=True).item()
        for k in quality_dict.keys():
            quality_dict[k] = quality_dict[k][n.newaxis]

        for i in range(1,len(self.metric_files)):
            new_data = n.load(self.metric_files[i],allow_pickle=True).item()
            for k in quality_dict.keys():
                quality_dict[k] = n.concatenate([quality_dict[k], new_data[k][n.newaxis]],axis=0)
        self.quality_dict = quality_dict


        res = self.job.load_registration_results()

        res['pc_peak_loc'] = n.concatenate(res['pc_peak_loc'],axis=0)
        res['int_shift'] = n.concatenate(res['int_shift'],axis=0) * self.job.params['voxel_size_um']
        res['sub_pixel_shifts'] = n.concatenate(res['sub_pixel_shifts'],axis=0) * self.job.params['voxel_size_um']
        res['phase_corr_max'] = n.concatenate([pcorr.max(axis=(1,2,3)) for pcorr in res['phase_corr_shifted']])
        self.res = res
        self.update_all_widgets()

    def update_plots(self, plane_idx = 0):

        fig = self.widgets['shifts']
        nt = len(self.res['phase_corr_max'])
        xs = n.arange(nt)
        if 'sub_pixel_shifts' not in self.lines.keys():
            l1 = fig.line(xs, self.res['sub_pixel_shifts'][:,0], 
                    line_width = self.plot_params['line_width'],
                    color=self.plot_params['colors'][2], 
                    legend_label = 'z', name = 'z')
            l2 = fig.line(xs, self.res['sub_pixel_shifts'][:,1], 
                    line_width = self.plot_params['line_width'],
                    color=self.plot_params['colors'][0], 
                    legend_label = 'y', name = 'y')
            l3 = fig.line(xs, self.res['sub_pixel_shifts'][:,2], 
                    line_width = self.plot_params['line_width'],
                    color=self.plot_params['colors'][1], 
                    legend_label = 'x', name = 'x')
            self.lines['sub_pixel_shifts'] = (l1,l2,l3)
        else:
            l1, l2, l3 = self.lines['sub_pixel_shifts']
            #TODO imlplement proper ColumnDataSource here
            l1.data_source = ColumnDataSource(dict(xs = xs, ys = self.res['sub_pixel_shifts'][:,0]))
            l2.data_source = ColumnDataSource(dict(xs = xs, ys = self.res['sub_pixel_shifts'][:,1]))
            l3.data_source = ColumnDataSource(dict(xs = xs, ys = self.res['sub_pixel_shifts'][:,2]))

        

        metrics = ['mean_fluorescence','volume_std', 'signal_range']

        fig = self.widgets['volume_quality_over_z']
        for i,metric in enumerate(metrics):
            ys = self.quality_dict[metric].mean(axis=0)
            xs = n.arange(ys.shape[0])
            
            if metric + '_z' not in self.lines.keys():
                self.lines[metric+'_z'] = fig.line(xs, ys, 
                        line_width = self.plot_params['line_width'],
                        color=self.plot_params['colors'][i],
                        legend_label=metric, name=metric)
            else:
                self.lines[metric+'_z'].data_source = ColumnDataSource(dict(xs = xs, ys = ys))
            

        fig = self.widgets['volume_quality_over_time']
        for i,metric in enumerate(metrics):
            print(metric)
            print(self.quality_dict[metric].shape)
            
            print(self.quality_dict[metric])
            ys = self.quality_dict[metric][:,plane_idx]
            xs = n.arange(ys.shape[0])
            if metric + '_t' not in self.lines.keys():
                self.lines[metric+'_t'] = fig.line(xs, ys, 
                        line_width = self.plot_params['line_width'],
                        color=self.plot_params['colors'][i],
                        legend_label=metric, name=metric)
            else:
                print("Updating line to %d" % plane_idx)
                self.lines[metric+'_t'].data_source = ColumnDataSource(dict(xs = xs, ys = ys))


    def update_all_widgets(self, plane_idx = 0):
        self.update_plane(plane_idx = plane_idx)
        self.update_plots(plane_idx = plane_idx)

    def update_plane(self, plane_idx = 0):
        if plane_idx < 0 or plane_idx >= self.nz:
            print("Invalid plane idx")
            return
        print("Rendering volume")
        vol = n.array([vol[plane_idx] for vol in self.imgs])
        self.widgets['mean_img_plane'].update_volume(vol)