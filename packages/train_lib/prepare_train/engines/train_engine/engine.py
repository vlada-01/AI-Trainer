import torch
import torch.optim as opt

from packages.train_lib.prepare_train.metrics.metrics_manager import MetricsManager
from packages.train_lib.prepare_train.loss.loss_manager import LossesManager

from packages.logger.logger import get_logger

log = get_logger(__name__)

def prepare_train_engine(train_meta):
    kwargs = {
        'log_train_metrics': train_meta.get('log_train_metrics'),
        'device': train_meta.get('device'),
        'epochs': train_meta.get('epochs'),
        'num_of_iters': train_meta.get('num_of_iters'),
        'optimizer': train_meta.get('optimizer'),
        'scheduler': train_meta.get('scheduler'),
        'losses': train_meta.get('losses'),
        'metrics': train_meta.get('metrics'),
    }
    return TrainEngine(**kwargs)

class TrainEngine:
    def __init__(self, device, epochs, num_of_iters, optimizer, scheduler, losses, metrics, log_train_metrics):
        self.device: str = device
        self.epochs: int = epochs
        self.num_of_iters: int = num_of_iters
        self.optimizer: opt.Optimizer= optimizer
        self.scheduler: opt.lr_scheduler.LRScheduler = scheduler
        self.losses: LossesManager = losses
        self.metrics: MetricsManager = metrics
        self.log_train_metrics: bool = log_train_metrics
    

    def train_model(self, model, train, val, writer):
        log.info('Starting model training')
        self.train_epochs(model, train, val, writer)

    def train_pp(self, model, val):
        if not model.are_pps_present():
            log.info('Post Processors are not present, skipping pp train')
            return None
        model.to(self.device)
        model.eval()
        for batch, _ in val:
            X, y = batch['X'], batch['y']
            X = {k: v.to(self.device) for k, v in X.items()}
            y = {k: v.to(self.device) for k, v in y.items()}

            logits = model.logits(X)

            model.collect_samples(logits, y)

        return model.fit_pps()

    def train_epochs(self, model, train, val, writer):
        model.to(self.device)
        for ep in range(self.epochs):
            log.info(f'Current Epoch: {ep}')
            self.train_epoch(model, train)

            log.info(f'Logging validation metrics for the epoch: {ep}')
            metrics_results, losses_results = self.eval_epoch(model, val)
            writer.log_metrics(metrics_results, 'validation', ep)
            writer.log_losses(losses_results, 'validation', ep)

            if self.log_train_metrics:
                log.info(f'Logging train metrics for the epoch: {ep}')
                metrics_results, losses_results = self.eval_epoch(model, train)
                writer.log_metrics(metrics_results, 'train', ep)
                writer.log_losses(losses_results, 'validation', ep)
           
            # TODO: does not work if the scheduler requires loss
            # does not work if the scheduler is batch based
            # maybe add some wrapper and inside it have different kinds of steps
            # engine calls it in all suitable places
            if self.scheduler is not None:
                self.scheduler.step()
        
            # TODO: maybe add checkpoints after each epoch
    
    def train_epoch(self, model, train):
        size = len(train.dataset)

        model.train()
        for i, (batch, indices) in enumerate(train):
            X, y = batch['X'], batch['y']
            X = {k: v.to(self.device) for k, v in X.items()}
            y = {k: v.to(self.device) for k, v in y.items()}
            for _ in range(self.num_of_iters):

                logits = model.logits(X)
                loss = self.losses.update(logits, y, detailed=False)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if i % 100 == 0:
                loss, current = loss.item(), (i + 1) * len(indices)
                log.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    def eval_epoch(self, model, dl):
        size = len(dl.dataset)

        self.losses.reset_losses()
        self.metrics.reset_metrics()

        model.eval()
        with torch.no_grad():
            for i, (batch, indices) in enumerate(dl):
                X, y = batch['X'], batch['y']
                X = {k: v.to(self.device) for k, v in X.items()}
                y = {k: v.to(self.device) for k, v in y.items()}
                
                logits = model.logits(X)
                h_outs = model.head_process(logits, apply_pp=False)

                loss = self.losses.update(logits, y, detailed=True)
                
                metrics_outs = model.get_metrics_outs(h_outs)
                self.metrics.update_metrics(metrics_outs, y)
                
                if i % 100 == 0:
                    loss, current = loss.item(), (i + 1) * len(indices)
                    log.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        losses_results = self.losses.collect_losses()
        metrics_results = self.metrics.collect_results()
        return metrics_results, losses_results