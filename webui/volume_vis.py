import numpy as n 
import os
from pathlib import Path

import param
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, RangeSlider
from bokeh.plotting import figure, show
from bokeh.models import LinearColorMapper
from bokeh.palettes import Turbo256,Greys256

class VolumeWidget(param.Parameterized):
    nz = param.Integer(6)
    ny = param.Integer(500)
    nx = param.Integer(500)
    def __init__(self, title, size=(600,600), volume=None):
        super().__init__()
        self.title = title
        self.size = size
        self.vmin = None
        self.vmax = None

        self.zidx = min(self.nz,5)
        if volume is None:
            volume = n.zeros((self.nz, self.ny, self.nx), float)
        self.volume = volume
        self.setup_figure()
        self.add_sliders()
        self.widget = column(self.plot, self.sliders['zidx'], self.sliders['image_range'],
                             name=self.title)


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
                                        end=self.volume.max() + 1e-5,
                                        step=0.01,
                                        value=(self.vmin, self.vmax),),
            'zidx' : Slider(title='Choose z-plane', start = 0, end = self.nz-1, step = 1, value = min(self.nz,5))
        }

        def update_limits(attrname, old ,new):
            self.vmin, self.vmax = self.sliders['image_range'].value
            self.mapper.low = self.vmin
            self.mapper.high = self.vmax
            print('New vlims:', self.vmin,self.vmax)

        def change_zidx(attrname, old ,new):
            self.zidx = self.sliders['zidx'].value
            print("New zidx:" , self.zidx)
            print('  STD: %.2f' % self.volume[self.zidx].std())
            print('  STD old : %.2f' % self.source.data['image'][0].std())
            
            self.source.data.update(image=[self.volume[self.zidx]])
            print('  STD new : %.2f' % self.source.data['image'][0].std())
            # self.source.data['image'] = self.volume[self.zidx]
            
            
        self.sliders['zidx'].on_change('value', change_zidx)
        self.sliders['image_range'].on_change('value',update_limits)

    def update_volume(self, new_volume):
        print("Vol update call!! ")
        self.volume = new_volume
        self.nz, self.ny, self.nx = self.volume.shape
        self.zidx = self.nz-1
        self.vmin = n.percentile(self.volume, 20)
        self.vmax = n.percentile(self.volume, 99.5)
        self.sliders['image_range'].start = self.volume.min()
        self.sliders['image_range'].end = self.volume.max() + 1e-5
        self.sliders['image_range'].value=(self.vmin, self.vmax)
        self.sliders['zidx'].start = 0
        self.sliders['zidx'].end = self.nz-1
        self.sliders['zidx'].value = self.zidx
        self.source.data.update(image = self.volume[self.zidx])
        print("Updated data!")
