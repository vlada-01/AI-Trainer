import mlflow
from .artifact_writer import ArtifactWriter

class MlflowWriter:
    def __init__(self, job_id, run_id):
        self.job_id = job_id
        self.run_id = run_id

        self.artifact_writer = None
    
    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, prefix='final', ep=None):
        for k, metrics_list in metrics.items():
            mlflow.log_metrics({f'{prefix.lower()}-{k}-{name.lower()}': metric_val for name, metric_val in metrics_list}, step=ep)

    def log_losses(self, losses: dict, prefix='final', ep=None):
        total_loss = losses['total_loss']
        raw_losses = losses['h_raw_losses']
        weighted_losses = losses['h_losses']
        mlflow.log_metric(f'{prefix.lower()}-total-loss', total_loss, ep)
        mlflow.log_metrics({f'{prefix.lower()}-{k}-raw-loss': v for k, v in raw_losses.items()}, step=ep)
        mlflow.log_metrics({f'{prefix.lower()}-{k}-weighted-loss': v for k, v in weighted_losses.items()}, step=ep)

    def open_artifact_writer(self):
        return ArtifactWriter(self.job_id, self.run_id)
    
    def set_tags(tags: dict):
        mlflow.set_tags(tags)