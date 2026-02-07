from enum import Enum
import sys
import torch
from pprint import pformat

from common.logger import get_logger

log = get_logger(__name__)

UNKNOWN_CLASS = -1

class AvailablePostProcessors(str, Enum):
    #for classification
    calibration = 'Calibration'
    global_threshold = 'GlobalThreshold'

    # for regressions
    # clipping, prediction_intervals, reject po nesigurnosti, rounding, linear_calibration

def build_post_processor(pp_cfg):
    log.debug('Initializing post processor builder for cfg:\n%s', pformat(pp_cfg.model_dump()))
    pp_list = []
    pp_names= []
    module = sys.modules[__name__]
    for pp in pp_cfg.post_processors:
        cls = getattr(module, pp.type, None)
        if cls is None:
            raise ValueError(f'There is no {pp.type} in AvailablePostProcessors')
        cfg_dict = pp.model_dump(exclude={'type', 'predefined'})
        pp_names.append(pp.type)
        pp = cls(**cfg_dict)
        log.info(f'Adding {type(pp).__name__} in Post Processor')
        pp_list.append(pp)
    log.info('Post Processor prepared successfully')
    return PostProcessor(pp_cfg, pp_list, pp_names)

# Current post processor always expects to have decision steps to already have preds
class PostProcessor():
    def __init__(self, pp_cfg,pp_list, pp_names):
        self.pp_cfg = pp_cfg
        self.pp_list = pp_cfg.post_processors
        self.pps = pp_list
        self.pp_names = pp_names
        self.trained_res = None

    def get_cfg(self):
        return self.pp_cfg

    def train(self, model, dl, device):
        model.set_pp(None)
        for p in model.parameters():
            p.requires_grad = False
        result = {}
        for i, pp in enumerate(self.pps):
            if pp.trainable:
                k, v_dict = pp.train(model, dl, device, result)
                result[k] = v_dict
                for k1, v1 in v_dict.items():
                    if hasattr(self.pp_list[i], k1):
                        setattr(self.pp_list[i], k1, v1)
                    else:
                        raise ValueError(f'Can not update field {k1} in {self.pp_list[i].type}. Available fields {self.pp_list[i].model_dump()}')
        self.trained_res = result
    
    def post_process(self, preds, error_table=None):
        pred_classes = None
        for name, pp in zip(self.pp_names, self.pps):
            if error_table is not None:
                preds, pred_classes = pp.process(preds, pred_classes)
                error_table[name.value] = preds.detach().cpu().numpy()
            else:
                preds, pred_classes = pp.process(preds, pred_classes)
        
        if error_table is not None:
            error_table['final'] = pred_classes.detach().cpu().numpy()
            return error_table, pred_classes
        return pred_classes
    
    def get_available_columns(self):
        return self.pp_names

class Calibration():
    def __init__(self, T=None, trainable=True):
        self.T = T
        self.trainable = trainable

    def train(self, model, dl, device, result):
        model.eval()

        T = torch.nn.Parameter(torch.ones(1, device=device))
        optimizer = torch.optim.Adam([T], lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        

        prev_T = 1.0
        patience = 2
        streak = 0
        eps = 1e-3

        for _ in range(3):
            for X, y, _ in dl:
                if isinstance(X, dict):
                    X = {k: v.to(device) for k, v in X.items()}
                else:
                    X = X.to(device)
                y = y.to(device)

                logits = model(X)

                loss = loss_fn(logits / T, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                curr_T = float(T.item())
                if abs(curr_T - prev_T) < eps:
                    streak += 1
                    if streak > patience:
                        self.T = float(T.detach().cpu().item())
                        log.info(f'Temperature converged earlier to {self.T}')
                        return AvailablePostProcessors.calibration, {'T': self.T}
                else:
                    streak = 0
                prev_T = curr_T

        self.T = float(T.detach().cpu().item())
        log.info(f'Learned temperature: {self.T}')
        return AvailablePostProcessors.calibration, {'T': self.T}

    def process(self, logits, pred_classes):
        probs = torch.softmax(logits / self.T, dim=1)
        conf, pred_classes = torch.max(probs, dim=1)
        return conf, pred_classes
    
class GlobalThreshold():
    def __init__(self, accuracy, threshold=None, trainable=True):
        self.accuracy = accuracy / 100

        self.threshold = threshold
        self.trainable = trainable

    def train(self, model, dl, device, result):
        if AvailablePostProcessors.calibration not in result:
            log.warning(f'Temperature Post Processor is ignored, fallback to T:1.0')
            T = 1.0
        else:
            T = result[AvailablePostProcessors.calibration]['T']
        
        val_conf = []
        val_correct = []
        
        model.eval()
        with torch.no_grad():
            for X, y, _ in dl:
                if isinstance(X, dict):
                    X = {k: v.to(device) for k, v in X.items()}
                else:
                    X = X.to(device)
                y = y.to(device)
                logits = model(X)
                preds = logits / T
                
                probs = torch.softmax(preds, dim=1)
                conf = probs.max(dim=1).values
                pred = probs.argmax(dim=1)
                correct = pred == y

                val_conf.append(conf.detach().cpu())
                val_correct.append(correct.detach().cpu())
        
        val_conf = torch.cat(val_conf)
        val_correct = torch.cat(val_correct)

        idx = torch.argsort(-val_conf)
        val_conf = val_conf[idx]
        val_correct = val_correct[idx]

        cum_acc = torch.cumsum(val_correct, dim=0) / (torch.arange(len(val_correct)) + 1)
        log.debug(f'GlobalThreshold {cum_acc}')

        valid = torch.where(cum_acc >= self.accuracy)[0]
        if len(valid) == 0:
            self.threshold = 1.0
        else:
            self.threshold = float(val_conf[valid[-1]])
        log.info(f'Global Threshold: {self.threshold}')
        accepted = val_conf >= self.threshold
        coverage = accepted.float().mean().item()
        log.info(f'Coverage: {coverage}')
        return AvailablePostProcessors.global_threshold, {'threshold': self.threshold}

    def process(self, preds, pred_classes):
        accepted = preds >= self.threshold
        preds_final = pred_classes.clone()
        preds_final[~accepted] = UNKNOWN_CLASS
        return preds, preds_final
