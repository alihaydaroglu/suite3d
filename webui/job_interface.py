import numpy as n
import os
from pathlib import Path

import panel as pn
import param

from suite3d.job import Job
from suite3d.job_registry import get_registered_jobs, touch_job

class JobInterface(param.Parameterized):
    job_loaded = param.Boolean(doc='Whether all files from job are loaded')
    def __init__(self, width = 200, height = 600):
        super().__init__()
        self.job_loaded = False
        self.jobdir_selector = pn.widgets.TextInput(\
                name = 'Suite3D Dir.', value = '/mnt/md0/runs')
        self.job_selector = pn.widgets.Select(name='Load job', options = [])
        self.load_job_button = pn.widgets.Button(name='Load existing job')

        # Registry-based job selector
        self._registered_jobs = get_registered_jobs()
        registry_options = {
            "%s  (%s)" % (j["job_id"], j["path"]): j["path"]
            for j in self._registered_jobs
        }
        self.registry_selector = pn.widgets.Select(
            name='Recent jobs', options=registry_options)
        self.load_registry_button = pn.widgets.Button(name='Load from registry')
        self.refresh_registry_button = pn.widgets.Button(name='Refresh')

        self.newjob_name_input = pn.widgets.TextInput(name = "Job name", value=' ')
        self.newjob_button = pn.widgets.Button(name = 'Create job')
        self.newjob_file_selector = pn.widgets.FileSelector('/',name='Select tiff dirs')

        self.loadjob_widgets = pn.WidgetBox(
            self.job_selector,
            self.load_job_button,
        )
        self.registry_widgets = pn.WidgetBox(
            self.registry_selector,
            pn.Row(self.load_registry_button, self.refresh_registry_button),
        )
        self.newjob_widgets = pn.WidgetBox(
            self.newjob_name_input,
            self.newjob_file_selector,
            self.newjob_button)

        self.job_widget = pn.Column(
            self.registry_widgets,
            pn.layout.Divider(),
            self.jobdir_selector, self.loadjob_widgets,
            width=width, height=height, name='Load job',)
        self.job = None
        self.job_data = {}

        self.bind_load_widgets()

    def bind_load_widgets(self):
        pn.bind(self.update_load_job_selector, dir_string=self.jobdir_selector, watch=True)
        self.update_load_job_selector(self.jobdir_selector.value)

        self.load_job_button.on_click(self.load_job)
        self.load_registry_button.on_click(self._load_from_registry)
        self.refresh_registry_button.on_click(self._refresh_registry)

    def _refresh_registry(self, event=None):
        self._registered_jobs = get_registered_jobs()
        self.registry_selector.options = {
            "%s  (%s)" % (j["job_id"], j["path"]): j["path"]
            for j in self._registered_jobs
        }

    def _load_from_registry(self, event):
        job_path = self.registry_selector.value
        if not job_path:
            return
        job_path = Path(job_path)
        dirname = job_path.name
        jobid = dirname[4:] if dirname.startswith("s3d-") else dirname
        rootdir = str(job_path.parent)

        touch_job(str(job_path))
        self._do_load_job(rootdir, jobid)

    def update_load_job_selector(self, dir_string):
        print(dir_string)
        print("Updating")
        if os.path.isdir(dir_string):
            self.job_selector.options = sorted(os.listdir(dir_string))
            self.jobdir_selector.name = 'Suite3D Dir (valid)'
        else:
            self.jobdir_selector.name = 'Suite3D Dir (invalid)'

    def load_job(self, event):
        rootdir = self.jobdir_selector.value
        jobid = self.job_selector.value
        if jobid[:4] == 's3d-':
            jobid = jobid[4:]
        self._do_load_job(rootdir, jobid)

    def _do_load_job(self, rootdir, jobid):
        print("loaded: ", self.job_loaded)
        self.job_loaded = False
        print("Loading %s %s" % (rootdir, jobid))
        jobdir = Path(rootdir) / ('s3d-' + jobid)
        job = Job(rootdir, jobid, create=False, params_path = jobdir / 'params.npy')
        job.update_root_path(rootdir)

        self.job = job
        self.job_data['jobdir'] = jobdir
        self.job_data['jobid'] = jobid
        self.job_data['summary'] = n.load(jobdir / 'summary' / 'summary.npy', allow_pickle=True).item()
        self.job_loaded = True
        print("Loaded job:", self.job_loaded)


    