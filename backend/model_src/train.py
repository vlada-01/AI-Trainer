import mlflow

from model_src.eval import evaluate

from common.logger import get_logger

log = get_logger(__name__)

def train_model(predictor, train, val, train_params):
    log_train_params(train_params.train_cfg)

    log_train_metrics = train_params.train_cfg.log_train_metrics
    
    predictor = start_train(predictor, train, val, train_params, log_train_metrics)
    return predictor

def start_train(predictor, train_dl, val_dl, train_params, log_train_metrics):
    log.info('Starting model training')
    predictor.get_model().to(train_params.device)
    for ep in range(train_params.epochs):
        log.info(f'Current Epoch: {ep}')
        predictor = train(predictor, train_dl, train_params)
        log.info(f'Logging val metrics for the epoch: {ep}')
        log_metrics(predictor, val_dl, train_params, 'val', ep)

        if log_train_metrics:
            log.info(f'Logging train metrics for the epoch: {ep}')
            log_metrics(predictor, train_dl, train_params, 'train', ep)
        # TODO: does not work if the scheduler requires loss
        if train_params.scheduler is not None:
            train_params.scheduler.step()
    
    return predictor

def train(predictor, dl, train_params):
    size = len(dl.dataset)

    predictor.get_model().train()
    device = train_params.device
    num_of_iters = train_params.num_of_iters
    loss_fn = train_params.loss_fn
    opt = train_params.optimizer
    
    for i, (batch, indices) in enumerate(dl):
        X, y = batch['X'], batch['y']
        X = {k: v.to(device) for k, v in X.items()}
        y = y.to(device)
        for _ in range(num_of_iters):
            logits = predictor.logits(X)
            loss = loss_fn(logits, y)

            loss.backward()
            opt.step()
            opt.zero_grad()

        if i % 100 == 0:
            loss, current = loss.item(), (i + 1) * len(indices)
            log.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return predictor

def log_train_params(train_cfg):
    log.info('Logging training params')
    mlflow.log_params(train_cfg.model_dump())

def log_metrics(predictor, dl, train_params, prefix, step=None):
    metrics, _ = evaluate(predictor, dl, train_params)
    mlflow.log_metrics({f'{prefix}_{name.lower()}': metric_val for name, metric_val in metrics}, step=step)

