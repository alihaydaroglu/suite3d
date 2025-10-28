
# Class to handle the actual job execution
class Worker:
    def __init__(self, job_id, job_data):
        self.job_id = job_id
        self.job_data = job_data
        # Initialize job-specific information (input files, processing settings, etc.)
        
    def run(self):
        """Execute the compute job (CPU/GPU processing)"""
        # Start the image processing task using provided data (handle CPU/GPU execution)
        
        pass

    def check_status(self):
        """Check the current status of the job (running, completed, failed)"""
        # Return the current status of the job
        pass

    def cancel(self):
        """Cancel a running job"""
        # Implement job cancellation (kill process or stop thread)
        pass

## create a Registration class that extends Worker and implements the registration workflow
class Registration(Worker):
    def __init__(self, job_id, job_data):
        super().__init__(job_id, job_data)
        self.registration_params = job_data['registration_params']
        # Initialize any additional registration-specific attributes

    def run(self):
        """Execute the registration job"""
        # Implement the registration logic here
        # This could involve calling functions from the registration module
        pass

    def check_status(self):
        """Check the current status of the registration job"""
        # Implement logic to check the status of the registration process
        pass

    def cancel(self):
        """Cancel a running registration job"""
        # Implement logic to cancel the registration process
        pass