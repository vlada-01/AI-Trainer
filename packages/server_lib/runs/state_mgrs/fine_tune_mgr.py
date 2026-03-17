from .state_mgr import StateManager, StateCode
from packages.server_lib.runs.run_ctx import AvailableRunTypes

class FineTuneStateManager(StateManager):
    def __init__(self):
        states = {
            StateCode.draft: {StateCode.prepare_default_run},
            StateCode.prepare_default_run: {StateCode.prepare_pp},
            StateCode.prepare_pp: {StateCode.final_eval},
            StateCode.final_eval: {StateCode.done, StateCode.failed},
            StateCode.done: set(),
            StateCode.failed: set()
        }
        super().__init__(AvailableRunTypes.base, states)