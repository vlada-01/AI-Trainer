import torch
import sys
from enum import Enum
from abc import ABC, abstractmethod

from model_src.models.post_processor import UNKNOWN_CLASS

from common.logger import get_logger

log = get_logger(__name__)

# TODO: there should be a support for coverage
# TODO: implement avg, weighted ... for show_metrics

class AvailableMetrics(str, Enum):
    # for classifications
    accuracy = 'Accuracy'
    precision = 'Precision'
    recall = 'Recall'
    f1_score = 'F1Score'
    # for regressions
    mse = 'MSE'
    mae = 'MAE'
    rmse = 'RMSE'
    r2 = 'R2'
    # for textuals
    bleu = 'Bleu'
    perplexity = 'Perplexity'
    
    # TODO: maybe add total exec time

def prepare_metrics(cfg_metrics, meta):
    metrics_dict = {}
    
    log.info('Preparing metrics dict')
    module = sys.modules[__name__]
    for metric_cfg in cfg_metrics:
        out_key = metric_cfg.out_key
        metrics = metric_cfg.metrics
        log.info(f'Assembling metrics for the out_key: {out_key}')
        assembled_metrics = []
        for metric in metrics:
            metric_cls = getattr(module, metric, None)

            if metric_cls is None:
                raise ValueError(f'{module} does not support metric {metric.value}')

            log.debug(f'Adding {metric.value} in the assembled metrics')
            uniques = meta.get_output_unique_values(out_key)
            assembled_metrics.append(metric_cls(uniques))
        metrics_dict[out_key] = assembled_metrics
    return metrics_dict

def reset_metrics(metrics_dict):
    for assembled_metrics in metrics_dict.values():
        for metric in assembled_metrics:
            metric.reset()

def update_metrics(metrics, preds, targets):
    for k, assembled_metrics in metrics.items():
        curr_preds = preds[k]
        curr_targets = targets[k]
        for metric in assembled_metrics:
            metric.update(curr_preds, curr_targets)

def show_results(metrics):
    results = {}
    for k, assembled_metrics in metrics.items():
        tmp = []
        for metric in assembled_metrics:
            tmp.append(metric.show())
        results[k] = tmp
    return results

#-----------------------------------------------------------------

class Metric(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, preds, targets, post_processor):
        pass

    @abstractmethod
    def show(self):
        pass

class Accuracy(Metric):
    def __init__(self, uniques):
        self.name = AvailableMetrics.accuracy
        self.scored = None
        self.ds_size = None

    def reset(self):
        self.scored = 0
        self.ds_size = 0
    
    def update(self, preds, targets):
        unknown_cls_mask = preds == UNKNOWN_CLASS
        preds = preds[~unknown_cls_mask]
        targets = targets[~unknown_cls_mask]

        self.ds_size += preds.size(0)
        self.scored += (preds == targets).type(torch.float).sum().item()

    def show(self):
        return self.name, self.scored * 100 / (self.ds_size + 1e-12)
    

class Precision(Metric):
    def __init__(self, uniques):
        self.name = AvailableMetrics.precision
        self.N =  uniques
        self.method = 'avg' #TODO: add weighted, macro i micro
        
        self.tps = None
        self.fps = None

    def reset(self):
        self.tps = torch.zeros(self.N, dtype=torch.float)
        self.fps = torch.zeros(self.N, dtype=torch.float)

    def update(self, preds, targets):
        unknown_cls_mask = preds == UNKNOWN_CLASS
        preds = preds[~unknown_cls_mask]
        targets = targets[~unknown_cls_mask]
        correct = (preds == targets)

        tps = torch.bincount(preds[correct], minlength=self.N)
        fps = torch.bincount(preds[~correct], minlength=self.N)

        self.tps += tps.cpu().type(torch.float)
        self.fps += fps.cpu().type(torch.float)

    def show(self):
        precisions = self.tps / (self.tps + self.fps).clamp_min(1)
        return self.name, 1 / self.N * torch.sum(precisions)
        
class  Recall(Metric):
    def __init__(self, uniques):
        self.name = AvailableMetrics.recall
        self.N = uniques

        self.tps = None
        self.fns = None

    def reset(self):
        self.tps = torch.zeros(self.N, dtype=torch.float)
        self.fns = torch.zeros(self.N, dtype=torch.float)
    
    def update(self, preds , targets):
        unknown_cls_mask = preds == UNKNOWN_CLASS
        preds = preds[~unknown_cls_mask]
        targets = targets[~unknown_cls_mask]
        correct = (preds == targets)

        tps = torch.bincount(preds[correct], minlength=self.N)
        fns = torch.bincount(targets[~correct], minlength=self.N)

        self.tps += tps.cpu().float()
        self.fns += fns.cpu().float()

    def show(self):
        recalls = self.tps / (self.tps + self.fns).clamp_min(1)
        return self.name, 1 / self.N * torch.sum(recalls)


class F1Score(Metric):
    def __init__(self, uniques):
        self.name = AvailableMetrics.f1_score
        self.N = uniques

        self.tps = None
        self.fps = None
        self.fns = None

    def reset(self):
        self.tps = torch.zeros(self.N, dtype=torch.float)
        self.fps = torch.zeros(self.N, dtype=torch.float)
        self.fns = torch.zeros(self.N, dtype=torch.float)

    def update(self, preds, targets):
        unknown_cls_mask = preds == UNKNOWN_CLASS
        preds = preds[~unknown_cls_mask]
        targets = targets[~unknown_cls_mask]
        correct = (preds == targets)

        tps = torch.bincount(preds[correct], minlength=self.N)
        fps = torch.bincount(preds[~correct], minlength=self.N)
        fns = torch.bincount(targets[~correct], minlength=self.N)

        self.tps += tps.cpu().float()
        self.fps += fps.cpu().float()
        self.fns += fns.cpu().float()

    def show(self):
        precisions = self.tps / (self.tps + self.fps).clamp_min(1)
        recalls = self.tps / (self.tps + self.fns).clamp_min(1)
        f1 = 2 / (1/precisions + 1/recalls).min_clamp(1e-12)
        return self.name, self.N * torch.sum(f1)