from .state_mgr import StateManager, StateCode
from ..run_types import AvailableRunTypes

class BaseRunStateManager(StateManager):
    def __init__(self):
        states = {
            StateCode.draft: {StateCode.prepare_ds, StateCode.prepare_default},
            StateCode.prepare_ds: {StateCode.prepare_model},
            StateCode.prepare_model: {StateCode.prepare_default},
            StateCode.prepare_default: {StateCode.training},
            StateCode.training: {StateCode.done, StateCode.failed},
            StateCode.done: set(),
            StateCode.failed: set()
        }
        super().__init__(AvailableRunTypes.base, states)
        