from bidict import bidict
from pymongo import MongoClient
from datetime import datetime
from flask import Flask, request
from bson import json_util, ObjectId
import os
from pathlib import Path
import sys
import numpy as n

sys.path.insert(0, ".")
print(os.getcwd())
manager_port = 8532
app_port = 8533
db_port = 27017
self_ip = "128.40.198.118"
default_analysis_dir = "/mnt/md0/runs"
DEBUG = True


from objects import Job, Dataset
from db import DatabaseService
from suite3d.job import Job as s3dDataset


class Suite3DInterface:
    def __init__(self, analysis_path, dataset_str, create=False):
        if analysis_path is None or analysis_path == "":
            analysis_path = default_analysis_dir
        self.jobdir = Path(analysis_path) / ("s3d-" + dataset_str)
        s3d = s3dDataset(analysis_path, dataset_str, params_path=self.jobdir / "params.npy", create=create)
        s3d.update_root_path(analysis_path)
        self.s3d = s3d
        self.summary_paths = {}
        self.load_init_pass()

    def load_init_pass(self):
        summary_exists = self.s3d.check_summary_exists()
        if not summary_exists:
            print("No S3D summary exists!")
            self.summary_paths = {}

        img_path = os.path.join(self.s3d.dirs["summary"], "ref_img_3d.npy")
        data_path = os.path.join(self.s3d.dirs["summary"], "data.npy")
        if not os.path.exists(img_path):
            print("Found summary.npy, splitting into smaller files")
            self.s3d.load_summary()
            n.save(img_path, self.s3d.summary["ref_img_3d"])
            n.save(
                data_path,
                {
                    "plane_shifts": self.s3d.summary["plane_shifts"],
                    "reference_info": self.s3d.summary.get("reference_info", None),
                },
            )
        print("Found smaller summary files")
        self.summary_paths = {"img_path": img_path, "data_path": data_path}


# Main Job Manager class that coordinates everything
class Manager:
    def __init__(self):
        self.db = DatabaseService(db_uri=f"mongodb://localhost:{db_port}/s3d")  # Initialize DB connection
        print(self.db.client.get_database())
        self.loaded_datasets = {}
        self.loaded_jobs = {}
        self.api_service = APIService(self)  # Initialize the API service

    def load_suite3d_dataset_from_id(self, dataset_id):
        dataset = self.db.get_dataset(dataset_id=dataset_id)
        # print(dataset_id)
        # print(dataset)
        return self.load_suite3d_dataset(dataset["analysis_path"], dataset["dataset_str"], dataset_id=dataset_id)

    def load_suite3d_dataset(self, analysis_path, dataset_str, dataset_id=None):
        s3d = Suite3DInterface(analysis_path, dataset_str=dataset_str)
        self.loaded_datasets[dataset_str] = s3d
        return s3d

    def create_dset(self, subject, date, experiments, analysis_path=None):
        dset = Dataset(subject, date, experiments, analysis_path=analysis_path)

        # this is not optimal - it's to handle the case where
        # you are creating a duplicate dset
        loaded_dset = self.db.get_dataset(dataset_str=dset.dataset_str, object=True)
        if loaded_dset is not None:
            print("Found existing dataset")
            dset = loaded_dset
            return None
        else:
            self.db.create_dataset(dset)
        return dset._id

    def create_job(self, job_type, params, dataset_id=None, dataset_str=None):
        print("Creating job")
        if dataset_id is None:
            dataset_id = self.db.get_dataset(dataset_str=dataset_str, object=False)["_id"]["$oid"]
        job = Job(job_type, params, status="created", dataset_id=dataset_id)
        self.active_jobs[job._id] = (job.dataset_id, job.job_type, job.created_at)
        self.db.create_job(job)
        self.db.add_job_to_dataset(dataset_id, job._id)
        return job._id

    def start(self):
        """Start the job manager service (API + job processing)"""
        self.api_service.start()  # Start the API service (Flask)


# Class for exposing API endpoints (e.g., using Flask)
class APIService:
    def __init__(self, job_manager: Manager):
        self.jm = job_manager  # Reference to the main job manager
        self.app = Flask(__name__)
        self.register_routes()

    def register_routes(self):
        """Register API routes."""

        @self.app.route("/api/datasets", methods=["POST"])
        def create_dataset():
            """Endpoint to create a new dataset."""
            data = request.json
            print("Attempting to create new dataset: ", data)
            dset_id = self.jm.create_dset(data["subject"], data["date"], data["experiments"])
            if dset_id == -1:
                return json_util.dumps({"message": "Dataset already exists", "id": str(dset_id)}), 202
            return json_util.dumps({"message": "Dataset created", "id": str(dset_id)}), 201

        @self.app.route("/api/jobs", methods=["POST"])
        def create_job():
            """Endpoint to create a new job."""
            data = request.json
            print("Attempting to create new job: ", data)
            job_id = self.jm.create_job(data["job_type"], None, dataset_str=data["dataset_str"])
            return json_util.dumps({"message": "Job created", "id": str(job_id)}), 201

        @self.app.route("/api/load_dataset", methods=["GET"])
        def load_suite3d_dataset():
            data = request.json
            print("Loading suite3D dataset")
            s3d = self.jm.load_suite3d_dataset_from_id(data["dataset_id"]["$oid"])
            print("Loaded dataset")
            return s3d.summary_paths, 200

        @self.app.route("/api/datasets", methods=["GET"])
        def get_datasets():
            print("getting datasets")
            return json_util.dumps((self.jm.db.get_datasets()))

        @self.app.route("/api/jobs", methods=["GET"])
        def get_jobs_from_dataset():
            # print("job request received, data:", request)
            data = request.json
            # print(f"looking for jobs with dataset id {data}")
            jobs = self.jm.db.get_jobs_with_dataset(dataset_id=ObjectId(data["dataset_id"]["$oid"]))
            # print("here are the found jobs")
            # print(jobs)
            return json_util.dumps(jobs)

        @self.app.route("/api/alljobs", methods=["GET"])
        def get_all_jobs():
            print("getting all jobs")
            jobs = self.jm.db.get_all_jobs()
            return json_util.dumps(jobs)

    def start(self, host="localhost", port=8532):
        """Start the Flask API server"""
        self.app.run(host=host, port=port, debug=DEBUG)


# Main entry point for the service
if __name__ == "__main__":
    job_manager = Manager()  # Initialize the job manager
    # job_manager.load_all_datasets()
    # job_manager.load_all_jobs()
    job_manager.start()  # Start the service (API + job processing loop)


# junk
"""
def load_all_datasets(self, max_num=None):
    loaded_dsets = self.db.get_datasets()
    if max_num is not None:
        loaded_dsets = loaded_dsets[:max_num]
    for dset in loaded_dsets:
        self.active_datasets[dset._id] = dset.dataset_str
    return loaded_dsets

def load_jobs_for_dataset(self, dataset_id):
    jobs = self.db.get_jobs_with_dataset(dataset_id=dataset_id)
    for job in jobs:
        self.active_jobs[job._id] = (job.dataset_id, job.job_type, job.created_at)
    return jobs
"""
