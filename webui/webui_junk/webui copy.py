import numpy as n 
import os
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

print("testing")
from suite3d.job import Job

class VolumeWidget:
    def __init__(self, volume, name, size=(600,600)):
        self.nz, self.ny, self.nx = volume.shape
        self.volume = volume
        self.name = name
        self.size = size

        self.zidx = min(self.nz,5)

        self.setup_figure()
        self.add_sliders()
        self.widget = column(self.plot, self.sliders['zidx'], self.sliders['image_range'])
    

    def setup_figure(self):
        self.vmin = n.percentile(self.volume, 20)
        self.vmax = n.percentile(self.volume, 99.9)


        self.source = ColumnDataSource(data=dict(image=[self.volume[self.zidx]]))
        self.plot = figure(height=self.size[0], width=self.size[1], title=self.name,
                    tools="crosshair,pan,reset,save,wheel_zoom",
                    active_scroll="wheel_zoom")
        self.plot.grid.grid_line_width = 0

        self.mapper = LinearColorMapper(palette=Greys256, low=self.vmin, high=self.vmax)
        self.plot.image(source=self.source,  x=0, y=0, dw=self.nx, dh=self.ny,color_mapper=self.mapper,
                    level="image")
        
    def add_sliders(self):
        self.sliders = {
            'image_range' : RangeSlider(title="Adjust color levels",
                                        start=self.volume.min(),
                                        end=self.volume.max(),
                                        step=0.01,
                                        value=(self.vmin, self.vmax),),
            'zidx' : Slider(title='Choose z-plane', start = 0, end = self.nz-1, step = 1, value = min(self.nz,5))
        }

        def update_limits(attrname, old ,new):
            vmin, vmax = self.sliders['image_range'].value
            self.mapper.low = vmin
            self.mapper.high = vmax
            # print(vmin,vmax)

        def change_zidx(attrname, old ,new):
            zidx = self.sliders['zidx'].value
            # print("New zidx:" , zidx)
            self.source.data.update(image=[self.volume[zidx]])
            
            
        self.sliders['zidx'].on_change('value', change_zidx)
        self.sliders['image_range'].on_change('value',update_limits)

    def update_volume(self, new_volume):
        self.volume = new_volume
        self.vmin = n.percentile(self.volume, 20)
        self.vmax = n.percentile(self.volume, 99.5)
        self.sliders['image_range'].start = self.volume.min()
        self.sliders['image_range'].end = self.volume.max()
        self.source.data.update(image = self.volume[self.zidx])

# class JobInteface:
def load_job(rootdir, jobid):
    print("Loading %s %s" % (rootdir, jobid))
    if jobid[:4] == 's3d-':
        jobid = jobid[4:]
    jobdir = Path(rootdir) / ('s3d-' + jobid)
    job = Job(rootdir, jobid, create=False, params_path = jobdir / 'params.npy')
    job.update_root_path(rootdir)
    vmap = n.load(jobdir / 'corrmap' / 'vmap.npy', allow_pickle=True)
    mean_img = n.load(jobdir / 'corrmap' / 'mean_img.npy', allow_pickle=True)
    summary = n.load(jobdir / 'summary' / 'summary.npy', allow_pickle=True).item()
    return vmap, mean_img, summary


pn.extension(design="fast")
dir_selector = pn.widgets.TextInput(name="Suite3D Directory", value='/mnt/md0/runs')
job_selector = pn.widgets.Select(name="Job Name", options=[])
job_load_button = pn.widgets.Button(name='Load Job!')


def get_jobs_in_dir(dir_string):
    print("test")
    print(dir_string)
    if os.path.isdir(dir_string):
        job_selector.options = os.listdir(dir_string)
        dir_selector.name = 'Suite3D Directory (valid)'
    else:
        dir_selector.name = 'Suite3D Directory (invalid)'
bound_dir_selector = pn.bind(get_jobs_in_dir, dir_string=dir_selector, watch=True)
get_jobs_in_dir(dir_selector.value)




selection_panel = pn.Column(dir_selector, job_selector, job_load_button)

ui = pn.Column(selection_panel)#, volume_vis_panel)

volwidget = None
def button_click(event):
    global volwidget
    root_dir = dir_selector.value
    job_name = job_selector.value
    vmap, mean_img, summary = load_job(root_dir, job_name)
    print("Loaded")
    if volwidget is None:
        print("Adding widget")
        volwidget = VolumeWidget(mean_img, 'Mean Image')
        ui.append(volwidget.widget)
job_load_button.on_click(button_click)

ui.servable()
# curdoc().add_root(row(file_input, vol_col))
# curdoc().add_root(vol_col)
# curdoc().title = "Suite3D"
