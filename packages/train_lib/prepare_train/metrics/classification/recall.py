from packages.train_lib.prepare_train.metrics.metrics_manager import Metric, AvailableMetrics
import torch

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