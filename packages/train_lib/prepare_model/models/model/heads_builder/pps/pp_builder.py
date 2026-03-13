from enum import Enum
from pprint import pformat
from abc import ABC, abstractmethod
import torch

from packages.train_lib.meta import AvailableTasks
from packages.prepare_model.models.model.heads_builder.pps.classification.calibration import Calibration
from packages.prepare_model.models.model.heads_builder.pps.classification.global_threshold import GlobalThreshold

from packages.logger.logger import get_logger

log = get_logger(__name__)

UNKNOWN_CLASS = -1

class AvailablePostProcessors(Enum):
    #for classification
    calibration = 0
    global_threshold = 1


PP_REGISTRY_MAP = {
    AvailableTasks.classification: {
        AvailablePostProcessors.calibration: Calibration,
        AvailablePostProcessors.global_threshold: GlobalThreshold,
    },
}

def attach_pps(heads, model_meta, pps_cfg):
    log.debug('Initializing post processor builder for cfg:\n%s', pformat(pps_cfg.model_dump()))
    if not pps_cfg:
        return heads
    specs_mapper = model_meta.specs_mapper
    pps_cfg = {specs_mapper(k): v for k, v in pps_cfg.items()}
    for k, head in heads.items():
        pps_list = pps_cfg[k]
        log.info(f'Adding PostProcessorChain in Head: {k}')
        pp_chain = build_pp(head.get_task, pps_list)
        head.attach_pps(pp_chain)
    log.info('PostProcessorChain(s) for Heads are successfully prepared')
    return heads

# TODO: add later support for more independent post processors, right now, only one chain pps are supported
def build_pp(task, pps_list):
    pps = []
    if task not in PP_REGISTRY_MAP:
        raise ValueError(f'Post Processor is not supported for the task: {task.value}')
    # safe check that user can bypass ordering problems
    # TODO: maybe later won't be easy to insert the pp at some point, new logic will be needed
    pps_list = sorted(pps_list, key=lambda x: x.type)
    for pp_cfg in pps_list:
        type = pp_cfg.type
        kwargs = pp_cfg.model_dump(exclude={'type'})
        if type not in PP_REGISTRY_MAP[task]:
            raise ValueError(f'Post Processor is not supported for the task, pp: {task, type}')
        log.info(f'Adding {type(type).__name__} in PostProcessorChain')
        pps.append(PP_REGISTRY_MAP[task][type](**kwargs))
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
                self.try_fallback(self, pp)

            self.state, details = pp.process(self.state, return_details)
            if details:
                detailed = {**detailed, **details}

        last_out_key = self.pps[-1].get_out_key()
        self.state['final'] = self.state.pop(last_out_key)
        return self.state, detailed
    
    def try_fallback(self, pp):
        log.debug(f'Running fallback for the pp" {pp.name}')
        self.state = pp.resolve(self.state)
    
    def collect_samples(self, logits, targets):
        self.state_buf['logits'].append(logits.detach().cpu())
        self.targets_buf.append(targets.detach().cpu())
        
    def fit_pps(self):
        state_buf = {k: torch.cat(v) for k, v in self.state_buf.items()}
        targets = torch.cat(self.targets_buf)
        for pp in self.pps:
            if pp.is_trainable():
                log.info(f'Initializing train for post processor: {pp.name}')
                pp.train(state_buf, targets)
                state_buf, _ = pp.process(state_buf, False)


class PostProcessor(ABC):
    def __init__(self, name, in_key, out_key, fallback_key, trainable=False):
        self.name = name
        self.in_key = in_key
        self.out_key = out_key
        self.trainable = trainable
        self.fallback_key = fallback_key

    def resolve(self, state):
        return state

    @abstractmethod
    def process(self, state, return_details=False):
        pass

    def is_trainable(self):
        return self.trainable
    
    @abstractmethod
    def train(self, model, val, device):
        pass

    def get_name(self):
        return self.name

    def get_in_key(self):
        return self.in_key

    def get_out_key(self):
        return self.out_key
    

