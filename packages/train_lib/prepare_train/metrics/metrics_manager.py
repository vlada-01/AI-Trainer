
from packages.train_lib.tasks import AvailableTasks

from .metric import AvailableMetrics
from .classification import Accuracy
from .classification import Precision
from .classification import Recall
from .classification import F1Score

from packages.logger import get_logger

log = get_logger(__name__)


# TODO: there should be a support for coverage
# TODO: implement avg, weighted ... for show_metrics



# TODO: update me
METRICS_REGISTRY_MAP = {
    AvailableTasks.classification: {
        AvailableMetrics.accuracy: Accuracy,
        AvailableMetrics.precision: Precision,
        AvailableMetrics.recall: Recall,
        AvailableMetrics.f1_score: F1Score
    },
    AvailableTasks.regression: {}
}

def prepare_metrics(metrics_cfg, meta):
    metrics_dict = {}
    log.info('Initializing Metrics')
    for k, metrics in metrics_cfg.items():
        log.info(f'Assembling metrics for the spec: {k}')
        task = meta.get_specs()[k].task
        if task not in METRICS_REGISTRY_MAP:
                raise ValueError(f'Metric is not supported for the spec: {task}')
        assembled_metrics = []
        for metric in metrics:
            log.debug(f'Adding {metric.value} in the assembled metrics')
            if metric not in METRICS_REGISTRY_MAP[task]:
                raise ValueError(f'Metric is not supported for the metric type: {metric}')
            metric_obj = METRICS_REGISTRY_MAP[task][metric]()
            metric_obj.set_states(meta, k)
            assembled_metrics.append(metric_obj)
        metrics_dict[k] = assembled_metrics

    return MetricsManager(metrics_dict)

class MetricsManager:
    def __init__(self, metrics_dict):
        self.metrics = metrics_dict

    def reset_metrics(self):
        for metrics in self.metrics.values():
            for metric in metrics:
                metric.reset()
    
    def update_metrics(self, h_outs, targets):
        for k, metrics in self.metrics.items():
            curr_h_outs = h_outs[k]
            curr_targets = targets[k]
            for metric in metrics:
                metric.update(curr_h_outs, curr_targets)

    def collect_results(self):
        results = {}
        for k, metrics in self.metrics.items():
            tmp = []
            for metric in metrics:
                tmp.append(metric.show())
            results[k] = tmp
        return results

