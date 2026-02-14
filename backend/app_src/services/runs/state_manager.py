from enum import Enum

class AvailableRunTypes(Enum):
    base = 'base'
    fine_tune = 'fine_tune'
    post_process = 'post_process'

class StateCode(Enum):
    draft = 0
    ds_prepared = 1
    model_prepared = 2
    train_params_prepared = 3
    cfg_ready = 4
    fine_tune_prepared = 5
    pp_prepared = 6
    running = 7
    final_eval = 8
    done = 9
    failed = 10

def get_base_run_states():
    BASE_RUN_STATES = {
        StateCode.draft: {StateCode.ds_prepared, StateCode.cfg_ready},
        StateCode.ds_prepared: {StateCode.model_prepared},
        StateCode.model_prepared: {StateCode.train_params_prepared},
        StateCode.train_params_prepared: {StateCode.cfg_ready},
        StateCode.cfg_ready: {StateCode.running},
        StateCode.running: {StateCode.final_eval, StateCode.failed},
        StateCode.final_eval: {StateCode.done},
        StateCode.failed: set()
    }
    return BASE_RUN_STATES

def get_fine_tune_states():
    FINE_TUNE_RUN_STATES = {
        StateCode.draft: {StateCode.cfg_ready},
        StateCode.fine_tune_prepared: {StateCode.running},
        StateCode.running: {StateCode.final_eval, StateCode.failed},
        StateCode.final_eval: {StateCode.done},
        StateCode.failed: set()
    }
    return FINE_TUNE_RUN_STATES

def get_post_process_run_states():
    POST_PROCESS_RUN_STATES = {
        StateCode.draft: {StateCode.cfg_ready},
        StateCode.pp_prepared: {StateCode.running},
        StateCode.running: {StateCode.final_eval, StateCode.failed},
        StateCode.final_eval: {StateCode.done},
        StateCode.failed: set()
    }
    return POST_PROCESS_RUN_STATES

RUN_TYPE_MAPPING = {
    AvailableRunTypes.base: get_base_run_states,
    AvailableRunTypes.fine_tune: get_fine_tune_states,
    AvailableRunTypes.post_process: get_post_process_run_states
}

def get_state_mappings(run_type):
    if run_type not in RUN_TYPE_MAPPING:
        raise ValueError(f'Run type "{run_type}" is not present in the RUN_TYPE_MAPPING')
    return RUN_TYPE_MAPPING[run_type](run_type)