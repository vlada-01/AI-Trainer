from pprint import pformat
import torch

from packages.train_lib.tasks import AvailableTasks

from .post_processor import AvailablePostProcessors
from .classification import Calibration
from .classification import GlobalThreshold

from packages.logger import get_logger

log = get_logger(__name__)


PP_REGISTRY_MAP = {
    AvailableTasks.classification: {
        AvailablePostProcessors.calibration: (Calibration, 0),
        AvailablePostProcessors.global_threshold: (GlobalThreshold, 1),
    },
}

def attach_pps(heads, pps_cfg):
    log.debug('Initializing post processor builder for cfg:\n%s', pformat(pps_cfg))
    if not pps_cfg:
        return heads
    for k, head in heads.items():
        pps_list = pps_cfg[k]
        log.info(f'Adding PostProcessorChain in Head: {k}')
        pp_chain = build_pp(head.get_task(), pps_list)
        head.attach_pps(pp_chain)
    log.info('PostProcessorChain(s) for Heads are successfully prepared')
    return heads

# TODO: add later support for more independent post processors, right now, only one chain pps are supported
def build_pp(task, pps_list):
    pps = []
    if task not in PP_REGISTRY_MAP:
        raise ValueError(f'Post Processor is not supported for the task: {task.value}')
    
    for pp_cfg in pps_list:
        pp_type = pp_cfg.type
        kwargs = pp_cfg.model_dump(exclude={'type'})
        if pp_type not in PP_REGISTRY_MAP[task]:
            raise ValueError(f'Post Processor is not supported for the task, pp: {task, pp_type}')
        log.info(f'Adding {type(pp_type).__name__} in PostProcessorChain')
        cls, priority = PP_REGISTRY_MAP[task][pp_type]
        pps.append((cls(**kwargs), priority))
    pps = sorted(pps, key=lambda x: x[1])
    pps = [t[0] for t in pps]
    return PostProcessorChain(pps)

class PostProcessorChain:
    def __init__(self, pps_list):
        self.pps = pps_list
        self.state = dict()

        self.state_buf = {'logits': []}
        self.targets_buf = []

    def post_process(self, x, return_details):
        self.state = {'logits': x}
        detailed = {}
        for pp in self.pps:
            in_key = pp.get_in_key()
            if in_key not in self.state:
                self.try_fallback(self, pp, self.state)

            self.state, details = pp.process(self.state, return_details)
            if details:
                detailed.update(details)

        last_out_key = self.pps[-1].get_out_key()
        self.state['final'] = self.state.pop(last_out_key)
        return self.state, detailed
    
    def try_fallback(self, pp, state):
        log.debug(f'Running fallback for the pp" {pp.name}')
        state = pp.resolve(state)
    
    def collect_samples(self, logits, targets):
        self.state_buf['logits'].append(logits.detach().cpu())
        self.targets_buf.append(targets.detach().cpu())
        
    def fit_pps(self):
        for_update = {}
        state_buf = {k: torch.cat(v) for k, v in self.state_buf.items()}
        targets = torch.cat(self.targets_buf)
        for pp in self.pps:
            if pp.is_trainable():
                log.info(f'Initializing train for post processor: {pp.name}')
                in_key = pp.get_in_key()
                if in_key not in self.state_buf:
                    self.try_fallback(pp, state_buf)
                    
                result = pp.train(state_buf, targets)
                for_update[pp.name] = result
                state_buf, _ = pp.process(state_buf, False)
        return for_update



    

