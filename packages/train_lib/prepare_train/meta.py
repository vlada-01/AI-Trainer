import torch
import torch.optim as opt
import inspect
from pprint import pformat

from packages.train_lib.prepare_train.metrics.metrics_manager import prepare_metrics, MetricsManager
from packages.train_lib.prepare_train.loss.loss_manager import prepare_losses, LossesManager
from packages.train_lib.prepare_train.error_tables.error_analysis_manager import prepare_error_analysis, ErrorAnalysisManager

from packages.logger.logger import get_logger

log = get_logger(__name__)

def create_meta(train_cfg, model_params, meta):
    log.debug('Preparing TrainMeta for cfg:\n%s', pformat(train_cfg.model_dump()))
    train_meta = TrainMeta(model_params, meta, train_cfg)
    log.debug('TrainMeta is successfully prepared')
    return train_meta

class TrainMeta:
    def __init__(self, model_params, meta, train_cfg):
        self.log_train_metrics: bool = train_cfg.log_train_metrics
        self.device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu") #TODO: needs to be changed if there are any other devices
        self.epochs: int = train_cfg.epochs
        self.num_of_iters: int = train_cfg.num_of_iters 
        self.optimizer: opt.Optimizer = self.prepare(opt, train_cfg.optimizer, model_params)
        self.scheduler: opt.lr_scheduler.LRScheduler = self.prepare(opt.lr_scheduler, train_cfg.lr_decay, self.optimizer) if train_cfg.lr_decay is not None else None
        self.losses: LossesManager = prepare_losses(train_cfg.loss_fns)
        self.metrics: MetricsManager = prepare_metrics(train_cfg.metrics, meta)
        self.error_analysis: ErrorAnalysisManager = prepare_error_analysis(meta)

    def get(self, attr_name):
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        raise ValueError(f'Train Meta does not have attr: {attr_name}')
    
    def to_dict(self):
        return {}
    #----------------------------------------

    def prepare(self, module, cfg, pos_param=None):
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
