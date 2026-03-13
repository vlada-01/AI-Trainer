import torch
import mlflow

from packages.logger.logger import get_logger

log = get_logger(__name__)

def prepare_train_engine(meta):
    kwargs = {
        'log_train_metrics': meta.get('log_train_metrics'),
        'device': meta.get('device'),
        'epochs': meta.get('epochs'),
        'num_of_iters': meta.get('num_of_iters'),
        'optimizer': meta.get('optimizer'),
        'scheduler': meta.get('scheduler'),
        'losses': meta.get('losses'),
        'metrics': meta.get('metrics'),
    }
    return TrainEngine(**kwargs)

class TrainEngine:
    def __init__(self, device, epochs, num_of_iters, optimizer, scheduler, losses, metrics, log_train_metrics):
        self.device = device
        self.epochs = epochs
        self.num_of_iters = num_of_iters
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.losses = losses
        self.metrics = metrics
        self.log_train_metrics = log_train_metrics
    

    def train_model(self, model, train, val):
        log.info('Starting model training')
        self.train_epochs(model, train, val)

    def train_pp(self, model, val):
        if not self.model.are_pps_present():
            log.info('Post Processors are not present, skipping pp train')
            return
        model.to(self.device)
        model.eval()
        for batch, _ in val:
            X, y = batch['X'], batch['y']
            X = {k: v.to(self.device) for k, v in X.items()}
            y = {k: v.to(self.device) for k, v in y.items()}

            logits = self.model.logits(X)

            self.model.collect_samples(logits, y)

        self.model.fit_pps()

    def train_epochs(self, model, train, val):
        model.to(self.device)
        model.train()
        for ep in range(self.epochs):
            log.info(f'Current Epoch: {ep}')
            self.train_epoch(model, train, val)

            log.info(f'Logging validation metrics for the epoch: {ep}')
            results = self.eval_epoch(val)
            self.log_metrics(results, ep, prefix='validation')

            if self.log_train_metrics:
                log.info(f'Logging train metrics for the epoch: {ep}')
                results = self.eval_epoch(train)
                self.log_metrics(results, ep, prefix='train')
           
            # TODO: does not work if the scheduler requires loss
            # does not work if the scheduler is batch based
            if self.scheduler is not None:
                self.scheduler.step()
        
            # TODO: maybe add checkpoints after each epoch
    
    def train_epoch(self, model, train):
        size = len(self.train.dataset)

        model.train()
        for i, (batch, indices) in enumerate(train):
            X, y = batch['X'], batch['y']
            X = {k: v.to(self.device) for k, v in X.items()}
            y = {k: v.to(self.device) for k, v in y.items()}
            for _ in range(self.num_of_iters):
                logits = self.model.logits(X)
                loss = self.losses.calculate_total_loss(logits, y)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if i % 100 == 0:
                loss, current = loss.item(), (i + 1) * len(indices)
                log.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    def eval_epoch(self, model, dl):
        size = len(dl.dataset)

        total_loss = 0
        self.metrics.reset_metrics()

        model.to(self.device)
        model.eval()
        with torch.no_grad():
            for i, (batch, indices) in enumerate(dl):
                X, y = batch['X'], batch['y']
                X = {k: v.to(self.device) for k, v in X.items()}
                y = {k: v.to(self.device) for k, v in y.items()}
                logits = model.logits(X)
                total_loss += self.losses.calculate_total_loss(logits, y).item()
                
                h_outs = model.head_process(logits, apply_pp=False)
                self.metrics.update_metrics(h_outs, y)
                
                if i % 100 == 0:
                    current = (i + 1) * len(indices)
                    log.info(f"loss: {total_loss:>7f}  [{current:>5d}/{size:>5d}]")

        metrics_results = self.metrics.collect_results()
        return metrics_results

    @staticmethod
    def log_metrics(results, ep, prefix):
        for k, metrics in results.items():
            mlflow.log_metrics({f'{prefix}/{k}/{name.lower()}': metric_val for name, metric_val in metrics}, step=ep)