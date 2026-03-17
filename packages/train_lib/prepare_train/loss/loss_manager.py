import inspect
import torch.nn as nn
from pprint import pformat

from packages.logger import get_logger

log = get_logger(__name__)

# supports only the losses from the nn module
def prepare_losses(loss_fns_cfg):
    log.info('Initializing Losses')
    loss_fns = {}
    loss_ws = {}
    for k, loss_fn in loss_fns_cfg.items():
        weight = loss_fn.weight
        fn = loss_fn.fn

        callable = getattr(nn, fn.type, None)
        if callable is None:
            raise ValueError(f'{nn.__name__} does not support callable {fn.type}')
        
        sig = inspect.signature(callable)
        allowed_params = sig.parameters
        kwargs = {}
        for k, v in fn.args.items():
            if k in allowed_params:
                kwargs[k] = v
            else:
                log.warning(f'{k} will be ignored for {nn.__name__}.{fn.type}')

        loss_fns[k] = callable(**kwargs)
        loss_ws[k] = weight

    log.debug(f'Initializing LossesManager with losses:\n%s', pformat(loss_fns))
    losses = LossesManager(loss_fns)
    log.info('Losses successfully prepared')
    return losses

class LossesManager:
    def __init__(self, loss_fns, loss_ws):
        self.loss_fns = loss_fns
        self.loss_ws = loss_ws
        self.h_weighted_losses = dict()
        self.h_raw_losses = dict()

    def reset_losses(self):
        self.h_weighted_losses = dict()
        self.h_raw_losses = dict()

    def update(self, logits, targets, detailed):
        total_loss = 0.0
        for k in self.loss_fns.keys():
            curr_logits = logits[k]
            curr_targets = targets[k]

            h_loss = self.loss_fns[k](curr_logits, curr_targets)
            h_w = self.loss_ws[k]
            if detailed:
                self.h_raw_losses[k] += h_loss
                self.h_weighted_losses[k] += h_loss * h_w

            total_loss += h_loss * h_w
        return total_loss
    
    def collect_losses(self):
        total_loss = sum([l for l in self.h_weighted_losses.values()])
        results = {
            'h_raw_losses': {k: v.item() for k, v in self.h_raw_losses.items()},
            'h_losses': {k: v.item() for k, v in self.h_weighted_losses.items()},
            'total_loss': total_loss.item()
            }
        return results
