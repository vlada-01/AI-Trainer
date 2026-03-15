from packages.train_lib.prepare_model.models.dag_net.dag_builder import build_dag
from packages.train_lib.prepare_model.models.model.heads_builder.heads_builder import build_heads
from packages.train_lib.prepare_model.models.model.heads_builder.pps.pp_builder import attach_pps

from packages.logger.logger import get_logger

log = get_logger(__name__)

def create_model(model_cfg, meta, model_meta):
    dag = build_dag(model_cfg.dag_cfg, model_meta)
    log.info('Initializing heads for Model')
    heads_dict = build_heads(meta)
    log.info('Adding Post Processors in Heads')
    heads_with_pp = attach_pps(heads_dict, model_meta, model_cfg.pps_cfg)
    model = Model(dag, heads_with_pp)
    log.info('Model successfully prepared')
    return model

def update_model_pps(model, meta, pps_cfg):
    log.info('Overriding current post processors in the model')
    heads_dict = model.get_heads()
    log.info('Adding new Post Processors in Heads')
    model_meta = meta.get_model_meta()
    heads_with_pp = attach_pps(heads_dict, model_meta, pps_cfg)
    model.set_heads(heads_with_pp)

class Model:
    def __init__(self, dag, heads_dict):
        self.dag = dag
        self.heads_dict = heads_dict
    
    def predict(self, x):
        self.to('cuda')
        self.eval()
        logits = self.logits(x)
        preds = self.head_process(self, logits)
        return preds

    def logits(self, x):
        return self.dag(x)
    
    # need to be extra careful to not overwrite logits
    def head_process(self, logits, apply_pp=True, return_details=False):
        return {k: head.process(logits[k], apply_pp, return_details) for k, head in self.heads_dict.items()}
    
    def get_metrics_outs(self, outs):
        return {k: head.get_metrics_out(outs[k]) for k, head in self.heads_dict.items()}
    
    def get_error_analysis_outs(self, outs):
        return outs
    
    def get_final_outs(self, outs):
        return {k: head.get_final_out(outs[k]) for k, head in self.heads_dict.items()}
    
    def are_pps_present(self):
        for head in self.heads_dict.values():
            if head.get_pps_chain():
                return True
        return False 
    
    def collect_samples(self, logits, targets):
        for k, head in self.heads_dict.items():
            head.collect_samples(logits[k], targets[k])
    
    def fit_pps(self):
        for head in self.heads_dict.values():
            head.fit_pps()
    
    def get_model(self):
        return self.dag
    
    def get_heads(self):
        return self.heads_dict
    
    def to(self, device):
        return self.dag.to(device)
    
    def eval(self):
        return self.dag.eval()
    
    def train(self):
        return self.dag.train()
    
    def state_dict(self):
        return self.dag.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.dag.load_state_dict(state_dict)

    def parameters(self):
        return self.dag.parameters()
    
    def device(self):
        return next(self.dag.parameters()).device
    
    def set_heads(self, new_heads_dict):
        self.heads_dict = new_heads_dict
    