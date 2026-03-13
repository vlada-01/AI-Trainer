from packages.train_lib.prepare_train.metrics.metrics_manager import Metric, AvailableMetrics
import torch

class F1Score(Metric):
    def __init__(self):
        self.name = AvailableMetrics.f1_score
        self.N = None

        self.tps = None
        self.fps = None
        self.fns = None

    def set_states(self, meta, key):
        data_meta = meta.get_data_meta()
        self.N = data_meta.get_output_unique_values(key)

    def reset(self):
        self.tps = torch.zeros(self.N, dtype=torch.float)
        self.fps = torch.zeros(self.N, dtype=torch.float)
        self.fns = torch.zeros(self.N, dtype=torch.float)

    def update(self, preds, targets):
        unknown_cls_mask = preds == -1
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