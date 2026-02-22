import inspect
import torch.nn as nn

from common.logger import get_logger

log = get_logger(__name__)

def prepare_loss_functions(loss_fns_cfg):
    loss_fns = {}
    for loss_fn in loss_fns_cfg:
        out_key = loss_fn.out_key
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

        loss_fns[out_key] = {
            'fn': callable(**kwargs),
            'w': weight
        }
    return Loss(loss_fns)

class Loss:
    def __init__(self, loss_fns):
        self.loss_fns = loss_fns
    
    def calculate_total_loss(self, logits, targets):
        total_loss = 0.0
        for k, v in self.loss_fns.items():
            curr_logits = logits[k]
            curr_targets = targets[k]
            Li = v['fn'](curr_logits, curr_targets)
            w = v['w']
            total_loss = total_loss + Li * w
        
        return total_loss