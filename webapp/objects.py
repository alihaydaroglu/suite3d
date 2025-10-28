from datetime import datetime
from bson import ObjectId

job_types = ["registration", "initial_pass", "sweep", "corrmap", "segmentation", "extraction"]

job_status = [
    "created",
    "running",
    "failed",
    "aborted",
    "completed",
    "deleted",
]

dataset_status = ["active", "archived", "deleted"]


class Dataset:
    fields = [
        "_id",
        "subject",
        "date",
        "experiments",
        "dataset_str",
        "jobs",
        "created_at",
        "metadata",
        "status",
        "analysis_path",
    ]

    def __init__(self, subject, date, experiments, analysis_path=None, metadata=None, status="active"):
        self._id = ObjectId()
        self.subject = subject
        self.date = date
        self.experiments = experiments
        self.make_dataset_str()
        self.status = "active"
        self.analysis_path = analysis_path

        self.jobs = []
        self.metadata = metadata if metadata is not None else {}
        self.created_at = datetime.now()

    @classmethod
    def from_dict(cls, data):
        for field in cls.fields:
            if field not in data.keys():
                print(f"Did not find field {field} in dset {data['dataset_str']}")
        sess = cls(data["subject"], data["date"], data["experiments"])
        for key in data.keys():
            setattr(sess, key, data[key])
        return sess

    def make_dataset_str(self):
        dataset_str = "%s_%s_" % (self.subject, self.date)
        for expn in self.experiments:
            dataset_str += "%s-" % str(expn)
        self.dataset_str = dataset_str[:-1]

    def to_dict(self):
        full_obj = vars(self)
        return full_obj


class Job:
    fields = [
        "_id",
        "job_type",
        "status",
        "dataset_id",
        "metadata",
        "params",
        "metadata",
        "created_at",
        "results",
        "log",
    ]

    def __init__(self, job_type, params, status, dataset_id, metadata=None):
        self._id = ObjectId()
        self.job_type = job_type
        self.status = status
        self.params = params
        self.dataset_id = dataset_id
        self.metadata = metadata if metadata is not None else {}
        self.created_at = datetime.now()
        self.results = []
        self.log = []

    def to_dict(self):
        full_obj = vars(self)
        return full_obj

    @classmethod
    def from_dict(cls, data):
        for field in cls.fields:
            if field not in data.keys():
                print(f"Did not find field {field} in job")
        sess = cls(data["job_type"], data["params"], data["status"], data["dataset_id"])
        for key in data.keys():
            setattr(sess, key, data[key])
        return sess


# class LogMessage():
#     def __init__(self, job_id, ):
