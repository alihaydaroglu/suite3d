import numpy as n 
import os
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


pn.extension(design="native")

job_interface = JobInterface(width=None, height=None)

job_widget_vis_button = job_interface.job_widget.controls(['visible'])[1]
job_widget_vis_button.name = 'Show create/load job widget'

summary_gridspec = pn.GridSpec(name='Initial Pass', sizing_mode='scale_both', max_height=800)
vol_widget = VolumeWidget(title='Reference Image')
summary_gridspec[:3, 0] = vol_widget.widget



plane_shift_figure = figure(width=400, height=400, sizing_mode='scale_both', title='Plane Shifts')
summary_gridspec[0, 1] = plane_shift_figure

xy_shift_figure = figure(width=400, height=400, sizing_mode='scale_both', title='Registration Shifts')
summary_gridspec[1,1] = xy_shift_figure

cmax_figure = figure(width=400, height=400, sizing_mode='scale_both', title='Registration quality')
summary_gridspec[2,1] = cmax_figure


ui = pn.Accordion(job_interface.job_widget, summary_gridspec)#, volume_vis_panel)

colors = ["#f38b0c","#d34ba1","#0fa4b8","#213073",'#8dda53']

def job_load_callback(value):
    if value:
        vol_widget.update_volume(job_interface.job_data['summary']['ref_img_3d'])
        nz = job_interface.job_data['summary']['plane_shifts'][:,0].shape[0]

        plane_shift_figure.line(n.arange(nz), job_interface.job_data['summary']['plane_shifts'][:,0], 
                                line_width = 3, legend_label='y', color=colors[0])
        plane_shift_figure.line(n.arange(nz), job_interface.job_data['summary']['plane_shifts'][:,1], 
                                line_width = 3, legend_label='x', color=colors[1])
        
        plane_shift_figure.legend.location = "top_left"
        plane_shift_figure.legend.click_policy="hide"
        plane_shift_figure.grid.grid_line_width = 0


        subpix_shifts = job_interface.job_data['summary']['reference_info']['subpix_shifts'][-1]
        nt = subpix_shifts.shape[0]
        
        xy_shift_figure.line(n.arange(nt), subpix_shifts[:,0], 
                                line_width = 3, legend_label='zshift', color=colors[2])
        xy_shift_figure.line(n.arange(nt), subpix_shifts[:,1], 
                                line_width = 3, legend_label='yshift', color=colors[0])
        xy_shift_figure.line(n.arange(nt), subpix_shifts[:,2], 
                                line_width = 3, legend_label='xshift', color=colors[1])
        xy_shift_figure.legend.location = "top_left"
        xy_shift_figure.legend.click_policy="hide"
        xy_shift_figure.grid.grid_line_width = 0

        summary = job_interface.job_data['summary']
        cmax = summary['reference_info']['cmax']
        frame_order = summary['reference_info']['frame_use_order']

        n_iter, n_frames = cmax.shape

        cmax_figure.line(n.arange(n_iter), cmax[:,(frame_order >= n_iter-1)].mean(axis=1),
                         line_width = 3, legend_label='phase corr', color=colors[3])

pn.bind(job_load_callback, job_interface.param.job_loaded, watch=True)

ui.servable()
