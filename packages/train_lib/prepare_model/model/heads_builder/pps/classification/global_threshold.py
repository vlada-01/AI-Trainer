import torch

from ..post_processor import AvailablePostProcessors
from ..post_processor import PostProcessor

from packages.logger.logger import get_logger

log = get_logger(__name__)

class GlobalThreshold(PostProcessor):
    def __init__(self, accuracy, threshold=None):
        super().__init__(name=AvailablePostProcessors.global_threshold,
                         in_key='probs',
                         out_key='preds',
                         fallback_key='logits',
                         trainable=True)
        self.accuracy = accuracy / 100

        self.threshold = threshold

    def train(self, state_buf, targets):
        logits = state_buf['logits']
        
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        val_conf = torch.max(probs, dim=1).values
        val_correct = pred == targets
        
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
        return {'threshold': self.threshold}

    def process(self, state, return_details):
        probs = state[self.in_key]
        preds = torch.argmax(probs, dim=1)
        conf = torch.max(probs, dim=1).values
        accepted = conf >= self.threshold
        preds[~accepted] = -1
        state[self.out_key] = preds
        if return_details:
            return state, {self.name: preds}
        return state, None
    
    def resolve(self, state):
        fallback_key = self.fallback_key
        logits = state[fallback_key]
        probs = torch.softmax(logits, dim=1)
        state['probs'] = probs
        return state