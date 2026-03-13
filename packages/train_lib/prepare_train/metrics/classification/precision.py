from packages.train_lib.prepare_train.metrics.metrics_manager import Metric, AvailableMetrics
import torch

from packages.core_lib.pps.post_processor import UNKNOWN_CLASS

class Precision(Metric):
    def __init__(self):
        self.name = AvailableMetrics.precision
        self.N = None
        self.method = 'avg' #TODO: add weighted, macro i micro
        
        self.tps = None
        self.fps = None

    def set_states(self, meta, key):
        data_meta = meta.get_data_meta()
        self.N = data_meta.get_output_unique_values(key)

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