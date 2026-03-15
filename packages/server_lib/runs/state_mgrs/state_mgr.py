from enum import Enum

class StateCode(Enum):
    draft = 'draft'
    prepare_ds = 'prepare dataset'
    prepare_model = 'prepare model'
    prepare_default = 'prepare default configuration'
    prepare_default_run = 'prepare default configuration from run'
    prepare_fine_tune = 'prepare fine-tune'
    prepare_pp = 'prepare post processor'
    training = 'training'
    final_eval = 'test evaluation'
    done = 'done'
    failed = 'failed'

class StateManager:
    def __init__(self, run_type, states):
        self.run_type = run_type
        self.states = states
        self.curr_state = StateCode.draft
    
    def is_valid_state(self, state_code):
        return state_code in self.state[self.curr_state]
    
    def move_state(self, new_state_code):
        self.curr_state = new_state_code

    def is_finished(self):
        return self.curr_state in (StateCode.done, StateCode.failed)
    
    def is_running(self):
        return self.curr_state in (StateCode.training, StateCode.final_eval)
    
