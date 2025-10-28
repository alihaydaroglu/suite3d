from pymongo import MongoClient
from datetime import datetime

from .worker_interface import Worker

# Class for managing active jobs
class JobList:
    def __init__(self):
        self.list = []  # Initialize a job list
        self.workers = {}  # Map of job_id to Worker instances

    def add_job(self, job_id, job_data):
        """Add a new job to the list"""
        # Add the job to the queue and map it to a worker
        pass

    def get_job_status(self, job_id):
        """Return the status of a specific job"""
        # Check the worker status for the job and return it
        pass

    def run_job(self, job_id):
        """Run a job"""
        # Start the job in a worker
        pass
    def abort_job(self, job_id):
        """Abort a job"""
        # Abort the job in the worker
        pass



# Class for interacting with MongoDB (or other database)
class DatabaseService:
    def __init__(self, db_connection):
        self.db = db_connection  # Initialize MongoDB connection
        
    def save_job_metadata(self, job_id, job_data):
        """Save initial job information to the database"""
        # Insert job information (status: queued) in MongoDB
        pass

    def update_job_status(self, job_id, status):
        """Update the status of a job in the database"""
        # Update the job status (running, completed, failed) in MongoDB
        pass

    def store_job_result(self, job_id, result_data):
        """Store the result of a job (e.g., image paths or analysis) in the database"""
        # Update MongoDB with result metadata (e.g., output file paths, thumbnails)
        pass

    def get_job_metadata(self, job_id):
        """Retrieve job metadata from the database"""
        # Fetch job information from MongoDB
        pass


# Class for exposing API endpoints (e.g., using Flask)
class APIService:
    def __init__(self, job_manager):
        self.job_manager = job_manager  # Reference to the main job manager
    
    def create_api(self):
        """Setup REST API routes for job management"""
        # Define API routes for job management (e.g., /submit_job, /job_status, /cancel_job)
        # Each route will call methods from JobManager
        pass

    def start(self):
        """Start the Flask API server"""
        # Start the Flask server to handle incoming requests
        pass


# Main Job Manager class that coordinates everything
class JobManager:
    def __init__(self):
        self.db_service = DatabaseService(db_connection="mongodb://localhost")  # Initialize DB connection
        self.job_list = JobList()  # Initialize the job list
        self.api_service = APIService(self)  # Initialize the API service

    def submit_job(self, job_data):
        """Submit a new job for execution"""
        job_id = self.generate_job_id()  # Generate a unique job ID
        self.db_service.save_job_metadata(job_id, job_data)  # Save job info to the database
        self.job_list.add_job(job_id, job_data)  # Add the job to the list
        return job_id  # Return the job ID to the user

    def check_job_status(self, job_id):
        """Check the status of a job"""
        status = self.job_list.get_job_status(job_id)  # Get the status from the job list
        return status

    def job_completed(self, job_id, result_data):
        """Handle the completion of a job"""
        self.db_service.update_job_status(job_id, "completed")  # Mark job as completed
        self.db_service.store_job_result(job_id, result_data)  # Store the result data
    
    def generate_job_id(self):
        """Generate a unique job ID"""
        # Create a unique job identifier (e.g., using a UUID or timestamp)
        pass

    def start(self):
        """Start the job manager service (API + job processing)"""
        self.api_service.create_api()  # Setup the API routes
        self.api_service.start()  # Start the API service (Flask)
        self.process_jobs()  # Begin processing jobs from the list
    


# Main entry point for the service
if __name__ == "__main__":
    job_manager = JobManager()  # Initialize the job manager
    job_manager.start()  # Start the service (API + job processing loop)
