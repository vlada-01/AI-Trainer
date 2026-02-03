import torch
import sys
from enum import Enum
from abc import ABC, abstractmethod

from model_src.models.post_processor import UNKNOWN_CLASS


# TODO: implement all classes from the enum
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
    
    # TODO: add total exec time

def prepare_metrics(cfg_metrics, meta):
    metrics = []
    module = sys.modules[__name__]
    for metric in cfg_metrics:
        metric_cls = getattr(module, metric, None)

        if metric_cls is None:
            raise ValueError(f'{module} does not support callable {metric}')

        metrics.append(metric_cls(meta))
    return metrics

def reset_metrics(metrics):
    for metric in metrics:
        metric.reset()

def update_metrics(metrics, preds, targets, post_processor=None):
    for metric in metrics:
        metric.update(preds, targets, post_processor)

def show_results(metrics):
    results = []
    for metric in metrics:
        results.append(metric.show())
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
    def __init__(self, meta):
        self.name = AvailableMetrics.accuracy
        self.scored = None
        self.ds_size = None

    def reset(self):
        self.scored = 0
        self.ds_size = 0
    
    def update(self, preds, targets, post_processor):
        self.ds_size += preds.size(0)
        if post_processor is not None:
            preds = post_processor.post_process(preds)
            unknown_cls_mask = preds == UNKNOWN_CLASS
            preds = preds[~unknown_cls_mask]
            targets = targets[~unknown_cls_mask]
        else:
            preds = preds.argmax(1)
        self.scored += (preds == targets).type(torch.float).sum().item()

    def show(self):
        return self.name, self.scored * 100 / self.ds_size
    

class Precision(Metric):
    def __init__(self, meta):
        self.name = AvailableMetrics.precision
        self.N = meta.get_unique_targets()
        self.method = 'avg' #TODO: add weighted, macro i micro
        
        self.tps = None
        self.fps = None

    def reset(self):
        self.tps = torch.zeros(self.N, dtype=torch.float)
        self.fps = torch.zeros(self.N, dtype=torch.float)

    def update(self, preds, targets, post_processor):
        if post_processor is not None:
            preds = post_processor.post_process(preds)
            unknown_cls_mask = preds == UNKNOWN_CLASS
            preds = preds[~unknown_cls_mask]
            targets = targets[~unknown_cls_mask]
        else:
            preds = preds.argmax(1)
        
        correct = (preds == targets)

        tps = torch.bincount(preds[correct], minlength=self.N)
        fps = torch.bincount(preds[~correct], minlength=self.N)

        self.tps += tps.cpu().type(torch.float)
        self.fps += fps.cpu().type(torch.float)

    def show(self):
        if self.method == 'avg':
            precisions = self.tps / (self.tps + self.fps).clamp_min(1)
            return self.name, 1 / self.N * torch.sum(precisions)
        
class  Recall(Metric):
    def __init__(self, meta):
        self.name = AvailableMetrics.recall
        self.N = meta.get_unique_targets()

        self.tps = None
        self.fns = None

    def reset(self):
        self.tps = torch.zeros(self.N, dtype=torch.float)
        self.fns = torch.zeros(self.N, dtype=torch.float)
    
    def update(self, preds , targets, post_processor):
        if post_processor is not None:
            preds = post_processor.post_process(preds)
            unknown_cls_mask = preds == UNKNOWN_CLASS
            preds = preds[~unknown_cls_mask]
            targets = targets[~unknown_cls_mask]
        else:
            preds = preds.argmax(1)

        correct = (preds == targets)

        tps = torch.bincount(preds[correct], minlength=self.N)
        fns = torch.bincount(targets[~correct], minlength=self.N)

        self.tps += tps.cpu().float()
        self.fns += fns.cpu().float()

    def show(self):
        recalls = self.tps / (self.tps + self.fns).clamp_min(1)
        return self.name, 1 / self.N * torch.sum(recalls)


class F1Score(Metric):
    def __init__(self, meta):
        self.name = AvailableMetrics.f1_score
        self.N = meta.get_unique_targets()

        self.tps = None
        self.fps = None
        self.fns = None

    def reset(self):
        self.tps = torch.zeros(self.N, dtype=torch.float)
        self.fps = torch.zeros(self.N, dtype=torch.float)
        self.fns = torch.zeros(self.N, dtype=torch.float)

    def update(self, preds, targets, post_processor):
        if post_processor is not None:
            preds = post_processor.post_process(preds)
            unknown_cls_mask = preds == UNKNOWN_CLASS
            preds = preds[~unknown_cls_mask]
            targets = targets[~unknown_cls_mask]
        else:
            preds = preds.argmax(1)

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