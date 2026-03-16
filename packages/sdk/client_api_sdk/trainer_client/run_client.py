import requests

# FIXME: need to add delete run, job from the internal state when needed
class RunClient:
    def __init__(self):
        self.runs = dict()

    def get_info(self, url):
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def start_run(self, url, payload):
        run = requests.post(url, json=payload.model_dump())
        run.raise_for_status()
        run_id = run.json()['run_id']
        self.runs[run_id] = set()
        return run_id
    
    def request_job(self, run_id, url, payload):
        job = requests.post(url, json=payload.model_dump())
        job.raise_for_status()
        job_id = job.json['id']
        self.runs[run_id] = job_id
        return job_id