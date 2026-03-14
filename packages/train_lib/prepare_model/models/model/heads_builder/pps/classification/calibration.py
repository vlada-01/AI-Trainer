import torch

from packages.train_lib.prepare_model.models.model.heads_builder.pps.post_processor import PostProcessor

from packages.logger.logger import get_logger

log = get_logger(__name__)

class Calibration(PostProcessor):
    def __init__(self, T=None):
        super().__init__(name='calibration',
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

        prev_T = 1.0
        patience = 2
        streak = 0
        eps = 1e-3

        for _ in range(5):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss_fn(logits / T, targets)

            curr_T = float(T.item())
            if abs(curr_T - prev_T) < eps:
                streak += 1
                if streak > patience:
                    self.T = float(T.detach().cpu().item())
                    log.info(f'Temperature converged earlier to {self.T}')
                    return 
            else:
                streak = 0
                prev_T = curr_T

        self.T = float(T.detach().cpu().item())
        log.info(f'Learned temperature: {self.T}')


    def process(self, state, return_details):
        logits = state[self.in_key]
        new_logits = logits / self.T
        state[self.out_key] = new_logits
        if return_details:
            probs = torch.softmax(new_logits, dim=1)
            conf = torch.argmax(probs, dim=1)
            return state, {self.name: conf}
        return state, None
