from pymongo import MongoClient
from objects import Job, Dataset
from bson import ObjectId

# https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/
# sudo systemctl restart mongod
#  sudo nano /etc/mongod.conf
# /var/log/mongodb/mongod.log


class DatabaseService:
    def __init__(self, db_uri, db_name="job_manager"):
        """Initialize MongoDB connection and set up collections"""
        self.client = MongoClient(db_uri)
        self.db = self.client[db_name]
        self.datasets = self.db["datasets"]
        self.jobs = self.db["jobs"]

    def create_job(self, job: Job):
        self.jobs.insert_one(job.to_dict())

    def create_dataset(self, dataset: Dataset, object=False):
        self.datasets.insert_one(dataset.to_dict())

    def add_job_to_dataset(self, dataset_id, job_id):
        self.datasets.update_one({"_id": dataset_id}, {"$push": {"jobs": job_id}})
        return

    def get_job(self, job_id, object=False):
        """Retrieve a job from the database and return it as a Job object."""
        job_data = self.jobs.find_one({"_id": job_id})
        if job_data:
            if not object:
                return job_data
            return Job.from_dict(job_data)
        else:
            raise ValueError(f"Job {job_id} not found")

    def get_jobs_with_dataset(self, dataset_id=None, dataset_str=None, object=False):
        if dataset_str is not None:
            dataset_id = self.get_dataset(dataset_str=dataset_str)["_id"]
        elif type(dataset_id) == str:
            dataset_id = ObjectId(dataset_id)
        jobs = list(self.jobs.find({"dataset_id": dataset_id}))
        print(f"Found {len(jobs)} jobs with dataset_id {dataset_id}")

        if len(jobs) > 0:
            # print(jobs[0])
            if object:
                return [Job.from_dict(j) for j in jobs]
            else:
                return [j for j in jobs]
        else:
            return []

    def get_all_jobs(self, object=False):
        all_jobs = []
        for dset in self.get_datasets():
            jobs = self.get_jobs_with_dataset(dataset_id=dset["_id"], object=object)
            all_jobs += jobs
        print(f"Found a total of {len(all_jobs)} jobs")
        return all_jobs

    def get_datasets(self, object=False):
        if not object:
            return self.datasets.find()
        return [Dataset.from_dict(ds) for ds in self.datasets.find()]

    def get_dataset(self, dataset_id=None, dataset_str=None, object=False):
        """Retrieve a dataset from the database and return it as a Dataset object."""
        if dataset_id is not None:
            dataset_data = self.datasets.find_one({"_id": ObjectId(dataset_id)})
        if dataset_str is not None:
            dataset_data = self.datasets.find_one({"dataset_str": dataset_str})
        if dataset_data:
            if object:

                return Dataset.from_dict(dataset_data)
            else:
                return dataset_data
        else:
            return None
            # raise ValueError(f"Dataset {dataset_id, dataset_str} not found")
