import torch

from ..metric import Metric, AvailableMetrics

class Accuracy(Metric):
    def __init__(self):
        self.name = AvailableMetrics.accuracy
        self.scored = None
        self.ds_size = None

    def set_states(self, meta, key):
        pass

    def reset(self):
        self.scored = 0
        self.ds_size = 0
    
    def update(self, preds, targets):
        unknown_cls_mask = preds == -1
        final_preds = preds[~unknown_cls_mask]
        final_targets = targets[~unknown_cls_mask]

        self.ds_size += final_preds.size(0)
        self.scored += (final_preds == final_targets).type(torch.float).sum().item()

    def show(self):
        return self.name, self.scored * 100 / (self.ds_size + 1e-12)