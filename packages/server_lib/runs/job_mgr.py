from datetime import datetime, timezone

def create_job_mgr():
    return JobManager()

class JobManager:
    def __init__(self):
        self.jobs = dict()
    
    def add_job(self, job):
        job_id = job.get_id()
        self.jobs[job_id] = job

    def update_job(self, job_id, **kwargs):
        if job_id not in self.jobs:
            raise RuntimeError(f'No job found when update job is required')
        self.jobs[job_id].update(**kwargs)

    def remove_jobs(self):
        now = datetime.now(timezone.utc)
        expired_ids = [
            job_id
            for job_id, job in self.jobs.items()
            if job.expired(now)
        ]
        for job_id in expired_ids:
            del self.jobs[job_id]

    def get_job(self, job_id):
        if job_id not in self.jobs:
            raise RuntimeError(f'No job found when get job is required')
        return self.jobs[job_id]

    def get_jobs(self):
        return [j for j in self.jobs.values()]
