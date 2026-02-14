import torch
import torch.nn as nn
import torch.optim as opt
import inspect
from dataclasses import dataclass
from typing import Sequence, Union
from pprint import pformat

from typing import Any

from model_src.prepare_train.metrics import prepare_metrics, Metric
from model_src.prepare_train.error_analysis import prepare_error_analysis, ClassificationErrorAnalysis, RegressionErrorAnalysis

from common.logger import get_logger

log = get_logger(__name__)

@dataclass
class TrainParams:
    train_cfg: Any
    device: str
    epochs: int
    num_of_iters: int
    optimizer: opt.Optimizer
    scheduler: opt.lr_scheduler.LRScheduler
    loss_fn: nn.modules.loss._Loss
    metrics: Sequence[Metric]
    error_analysis: Union[ClassificationErrorAnalysis, RegressionErrorAnalysis]

def prepare_train_params(model_params, meta, train_cfg):
    log.debug('Preparing train params for cfg:\n%s', pformat(train_cfg.model_dump()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #TODO: needs to be changed if there are any other devices
    epochs = train_cfg.epochs
    num_of_iters = train_cfg.num_of_iters
    optimizer = prepare(opt, train_cfg.optimizer, model_params)
    scheduler = prepare(opt.lr_scheduler, train_cfg.lr_decay, optimizer) if train_cfg.lr_decay is not None else None
    loss_fn = prepare(nn, train_cfg.loss_fn)
    metrics = prepare_metrics(train_cfg.metrics, meta)
    error_analysis = prepare_error_analysis(meta)

    log.debug('Train Params are successfully prepared')
    return TrainParams(
        train_cfg=train_cfg,
        device=device,
        epochs=epochs,
        num_of_iters=num_of_iters,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        metrics=metrics,
        error_analysis=error_analysis
    )
    

def prepare(module, cfg, pos_param=None):
    callable = getattr(module, cfg.type, None)
    if callable is None:
        raise ValueError(f'{module} does not support callable {cfg.type}')
    
    sig = inspect.signature(callable)
    allowed_params = sig.parameters
    kwargs = {}
    for k, v in cfg.args.items():
        if k in allowed_params:
            kwargs[k] = v
        else:
            log.warning(f'{k} will be ignored for {module}.{cfg.type}')
    return callable(pos_param, **kwargs) if pos_param is not None else callable(**kwargs)

def update_train_cfg(new_train_cfg, old_train_cfg):
    log.debug('Updating train params for cfg:\n%s', pformat(new_train_cfg.model_dump()))
    for k in new_train_cfg.model_fields:
        if hasattr(old_train_cfg.train_cfg, k):
            v = getattr(new_train_cfg, k)
            setattr(old_train_cfg.train_cfg, k, v)
        else:
            raise ValueError(f'TrainCfg does not have {k} as the field name')
    return old_train_cfg
    