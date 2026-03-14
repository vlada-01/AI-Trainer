import mlflow
from packages.mlflow_logger.artifact_writer import ArtifactWriter

class MlflowWriter:
    def __init__(self, job_id, run_id):
        self.job_id = job_id
        self.run_id = run_id

        self.artifact_writer = None
    
    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, prefix='final', ep=None):
        for k, metrics_list in metrics.items():
            self.log_metrics({f'{prefix.lower()}-{k}-{name.lower()}': metric_val for name, metric_val in metrics_list}, step=ep)

    def open_artifact_writer(self):
        return ArtifactWriter(self.job_id, self.run_id)
    
    def set_tags(tags: dict):
        mlflow.set_tags(tags)