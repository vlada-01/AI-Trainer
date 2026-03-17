import torch

from .base_head import Head

class Classification(Head):
    def __init__(self, task):
        super().__init__(task=task)

    def process(self, logits, apply_pp, return_details):
        if not apply_pp:
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            return {
                'mandatory': {
                    'probs': probs,
                    'final': preds
                },
                'optional': {}
            }
        else:
            results, details = self.pps_chain.post_process(logits, return_details)
            return {
                'mandatory': {
                    'probs': results['probs'],
                    'final': results['final']
                },
                'optional': details
            }
    
    def get_final_out(self, h_out):
        return h_out['mandatory']['final']

    def get_metrics_out(self, h_out):
        return h_out['mandatory']['final']
    
    def get_error_analysis_out(self, h_out):
        return h_out