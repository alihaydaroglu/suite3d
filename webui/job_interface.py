import sys
import numpy as n 
import os
from pathlib import Path

import panel as pn
import param

from suite3d.job import Job

class JobInterface(param.Parameterized):
    job_loaded = param.Boolean(doc='Whether all files from job are loaded')
    def __init__(self, width = 200, height = 600):
        super().__init__()
        self.job_loaded = False
        self.jobdir_selector = pn.widgets.TextInput(\
                name = 'Suite3D Dir.', value = '/mnt/md0/runs')
        self.job_selector = pn.widgets.Select(name='Load job', options = [])
        self.load_job_button = pn.widgets.Button(name='Load existing job')
        
        self.newjob_name_input = pn.widgets.TextInput(name = "Job name", value=' ')
        self.newjob_button = pn.widgets.Button(name = 'Create job')
        self.newjob_file_selector = pn.widgets.FileSelector('/',name='Select tiff dirs')

        self.loadjob_widgets = pn.WidgetBox(
            self.job_selector,
            self.load_job_button,
        )
        self.newjob_widgets = pn.WidgetBox(
            self.newjob_name_input,
            self.newjob_file_selector,
            self.newjob_button)

        self.job_tabs = pn.Accordion( ("Create job", self.newjob_widgets))
        
        self.job_widget = pn.Column(self.jobdir_selector, self.loadjob_widgets, self.job_tabs,
                                    width=width, height=height,name='Create or load job',)
        self.job = None
        self.job_data = {}

        self.bind_load_widgets()
        
    def bind_load_widgets(self):
        pn.bind(self.update_load_job_selector, self=self, dir_string=self.jobdir_selector, watch=True)
        self.update_load_job_selector(self.jobdir_selector.value)

        self.load_job_button.on_click(self.load_job)

    def update_load_job_selector(self, dir_string):
        print(dir_string)
        print("Updating")
        if os.path.isdir(dir_string):
            self.job_selector.options = os.listdir(dir_string)
            self.jobdir_selector.name = 'Suite3D Dir (valid)'
        else:
            self.jobdir_selector.name = 'Suite3D Dir (invalid)'

    def load_job(self, event):
        print("loaded: ", self.job_loaded)
        self.job_loaded=False
        rootdir = self.jobdir_selector.value
        jobid = self.job_selector.value
        print("Loading %s %s" % (rootdir, jobid))
        if jobid[:4] == 's3d-':
            jobid = jobid[4:]
        jobdir = Path(rootdir) / ('s3d-' + jobid)
        job = Job(rootdir, jobid, create=False, params_path = jobdir / 'params.npy')
        job.update_root_path(rootdir)

        self.job = job
        self.job_data['jobdir'] = jobdir
        self.job_data['jobid'] = jobid
        # self.job_data['vmap'] = n.load(jobdir / 'corrmap' / 'vmap.npy', allow_pickle=True)
        # self.job_data['mean_img'] = n.load(jobdir / 'corrmap' / 'mean_img.npy', allow_pickle=True)
        # self.job_data['max_img'] = n.load(jobdir / 'corrmap' / 'max_img.npy', allow_pickle=True)
        self.job_data['summary'] = n.load(jobdir / 'summary' / 'summary.npy', allow_pickle=True).item()
        self.job_loaded = True
        print("Loaded job:", self.job_loaded)


    