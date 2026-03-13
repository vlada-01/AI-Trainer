import torch

from packages.logger.logger import get_logger

log = get_logger(__name__)

def prepare_eval_engine(meta):
    kwargs = {
        'device': meta.get('device'),
        'losses': meta.get('losses'),
        'metrics': meta.get('metrics'),
        'error_analysis': meta.get('error_analysis'),
    }
    return EvaluationEngine(**kwargs)

class EvaluationEngine:
    def __init__(self, device, losses, metrics, error_analysis):
        self.device = device
        self.losses = losses
        self.metrics = metrics
        self.error_analysis = error_analysis

    def evaluate(self, model, dl, return_details=False):
        size = len(self.dl.dataset)

        loss = 0
        self.metrics.reset_metrics()
        self.error_analysis.restart_error_tables()
        
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            for i, (batch, indices) in enumerate(dl):
                X, y = batch['X'], batch['y']
                X = {k: v.to(self.device) for k, v in X.items()}
                y = {k: v.to(self.device) for k, v in y.items()}
                logits = model.logits(X)
                
                loss += self.losses.compute_total_loss(logits, y).item()
                
                h_outs = model.head_process(logits, return_details)
                self.error_analysis.update_error_tables(indices, h_outs, y)
                
                self.metrics.update_metrics(h_outs, y)
                
                if i % 100 == 0:
                    current = (i + 1) * len(indices)
                    log.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        metrics_results = self.metrics.collect_results()
        dict_error_tables = self.error_analysis.get_results()
        #TODO: maybe add losses per head
        # metric_results['loss'] = loss
        return metrics_results, dict_error_tables
