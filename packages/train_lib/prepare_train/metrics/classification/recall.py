import torch

from ..metric import Metric, AvailableMetrics

class  Recall(Metric):
    def __init__(self):
        self.name = AvailableMetrics.recall
        self.N = None

        self.tps = None
        self.fns = None

    def set_states(self, meta, key):
        data_meta = meta.get_data_meta()
        self.N = data_meta.get_output_unique_values(key)

    def reset(self):
        self.tps = torch.zeros(self.N, dtype=torch.float)
        self.fns = torch.zeros(self.N, dtype=torch.float)
    
    def update(self, preds , targets):
        unknown_cls_mask = preds == -1
        final_preds = preds[~unknown_cls_mask]
        final_targets = targets[~unknown_cls_mask]
        correct = (final_preds == final_targets)

        tps = torch.bincount(final_preds[correct], minlength=self.N)
        fns = torch.bincount(final_targets[~correct], minlength=self.N)

        self.tps += tps.cpu().float()
        self.fns += fns.cpu().float()

    def show(self):
        recalls = self.tps / (self.tps + self.fns).clamp_min(1)
        return self.name, 1 / self.N * torch.sum(recalls)