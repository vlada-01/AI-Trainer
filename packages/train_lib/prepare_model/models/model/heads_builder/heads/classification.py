import torch

from packages.train_lib.prepare_model.models.model.heads_builder.heads.base_head import Head

class Classification(Head):
    def __init__(self, task):
        super().__init__(task=task)

    def process(self, x, apply_pp, return_details):
        if not apply_pp:
            probs = torch.softmax(x, dim=1)
            preds = probs.argmax(dim=1)
            return {
                'mandatory': {
                    'probs': probs,
                    'final': preds
                },
                'optional': {}
            }
        else:
            results, details = self.pps_chain.post_process(x, return_details)
            return {
                'mandatory': {
                    'probs': results['probs'],
                    'final': results['final']
                },
                'optional': details
            }