import mlflow

from model_src.prepare_train.prepare_train import prepare_train_params
from model_src.eval import evaluate

from common.logger import get_logger

log = get_logger(__name__)

def train_model(model, train, val, meta, cfg):
    train_cfg = cfg.train_cfg
    log.info(f'Preparing Training Params with {train_cfg}')
    train_params = prepare_train_params(model.parameters(), meta, train_cfg)
    log_train_params(train_cfg)

    log_train_metrics = cfg.log_train_metrics
    
    model = start_train(model, train, val, train_params, log_train_metrics)
    return model

def start_train(model, train_dl, val_dl, train_params, log_train_metrics):
    model = model.to(train_params.device)
    log.info('Starting model training')
    for ep in range(train_params.epochs):
        log.info(f'Current Epoch: {ep+1}')
        model = train(model, train_dl, train_params)
        log.info(f'Logging val metrics for the epoch: {ep}')
        log_metrics(model, val_dl, train_params, 'val', ep)

        if log_train_metrics:
            log.info(f'Logging train metrics for the epoch: {ep}')
            log_metrics(model, train_dl, train_params, 'train', ep)
        if train_params.scheduler is not None:
            train_params.scheduler.step()
    # log.info(f'Logging final metrics for the epoch')
    # log_metrics(model, val_dl, train_params, 'final_val')
    
    return model

def train(model, dl, train_params):
    size = len(dl.dataset)
    model.train()
    device = train_params.device
    num_of_iters = train_params.num_of_iters
    loss_fn = train_params.loss_fn
    opt = train_params.optimizer
    for batch, (X, y, _) in enumerate(dl):
        X, y = X.to(device), y.to(device)
        for _ in range(num_of_iters):
            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            opt.step()
            opt.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return model

def log_train_params(train_cfg):
    log.info('Logging training params')
    mlflow.log_params(train_cfg.model_dump())

def log_metrics(model, dl, train_params, prefix, step=None):
    metrics, _ = evaluate(model, dl, train_params)
    mlflow.log_metrics({f'{prefix}_{name.lower()}': metric_val for name, metric_val in metrics}, step=step)

