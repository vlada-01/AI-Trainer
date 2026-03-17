from ..run_types import AvailableRunTypes

from .base_run_mgr import BaseRunStateManager
from .pp_run_mgr import PPRunStateManager
from .fine_tune_mgr import FineTuneStateManager
from .final_eval_mgr import FinalEvalStateManager

STATE_MGR_REGISTRY_MAP = {
    AvailableRunTypes.base: BaseRunStateManager,
    AvailableRunTypes.post_process: PPRunStateManager,
    AvailableRunTypes.fine_tune: FineTuneStateManager,
    AvailableRunTypes.final_evaluation: FinalEvalStateManager
}

def create_state_mgr(run_type):
    if run_type not in STATE_MGR_REGISTRY_MAP:
        raise ValueError(f'There is no supported State Manager for given run type: {run_type.value}')
    return STATE_MGR_REGISTRY_MAP[run_type]()