import torch

from ..post_processor import AvailablePostProcessors
from ..post_processor import PostProcessor

from packages.logger import get_logger

log = get_logger(__name__)

class Calibration(PostProcessor):
    def __init__(self, T=None):
        super().__init__(name=AvailablePostProcessors.calibration,
                         in_key='logits',
                         out_key='logits',
                         fallback_key='logits',
                         trainable=True)
        self.T = T

    def train(self, state_buf, targets):
        logits = state_buf['logits']
        T = torch.nn.Parameter(torch.ones(1, device='cpu'))
        optimizer = torch.optim.Adam([T], lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        epochs = 5
        prev_loss = None
        eps = 1e-4

        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(logits / T, targets)
            
            loss.backward()
            optimizer.step()

            curr_loss = loss.item()
            if prev_loss is None:
                prev_loss = curr_loss
                continue

            diff = abs(prev_loss - curr_loss)
            if diff < eps:
                self.T = float(T.detach().cpu().item())
                log.info(f'Temperature converged earlier to {self.T}')
                return 
            log.info(f'Abs differnece in prev and current loss: {diff}')

        self.T = float(T.detach().cpu().item())
        log.info(f'Learned temperature: {self.T}')
        return {'T': self.T}


    def process(self, state, return_details):
        logits = state[self.in_key]
        new_logits = logits / self.T
        state[self.out_key] = new_logits
        if return_details:
            probs = torch.softmax(new_logits, dim=1)
            conf = torch.argmax(probs, dim=1)
            return state, {self.name: conf}
        return state, None
