from .artifact_reader import ArtifactReader

class MlflowReader:
    def __init__(self, job_id, run_id):
        self.job_id = job_id
        self.run_id = run_id

        self.artifact_reader = None

    def open_artifact_reader(self):
        return ArtifactReader(self.job_id, self.run_id)
    
    # FIXME: implement this fucnction
    def load_metrics(self):
        pass