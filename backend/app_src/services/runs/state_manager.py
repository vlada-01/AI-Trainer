from enum import Enum

class AvailableRunTypes(Enum):
    base = 'base'
    fine_tune = 'fine_tune'
    post_process = 'post_process'
    # final_evaluation = 'final_evaluation'

# TODO: need to update this crap, final_eval, pp, and fine tune

class StateCode(Enum):
    draft = 0
    prepare_ds = 1
    prepare_model = 2
    prepare_default = 3
    prepare_default_run = 4
    prepare_fine_tune = 5
    prepare_pp = 6
    training = 7
    final_eval = 8
    done = 9
    failed = 10

def get_base_run_states():
    BASE_RUN_STATES = {
        StateCode.draft: {StateCode.prepare_ds, StateCode.prepare_default},
        StateCode.prepare_ds: {StateCode.prepare_model},
        StateCode.prepare_model: {StateCode.prepare_default},
        StateCode.prepare_default: {StateCode.training},
        StateCode.training: {StateCode.done, StateCode.failed},
        StateCode.done: set(),
        StateCode.failed: set()
    }
    return BASE_RUN_STATES

# TODO: these two should first load from run and then update
def get_fine_tune_states():
    FINE_TUNE_RUN_STATES = {
        StateCode.draft: {StateCode.prepare_default_run},
        StateCode.prepare_default_run: {StateCode.prepare_fine_tune},
        StateCode.prepare_fine_tune: {StateCode.training},
        StateCode.training: {StateCode.done, StateCode.failed},
        StateCode.done: set(),
        StateCode.failed: set()
    }
    return FINE_TUNE_RUN_STATES

def get_post_process_run_states():
    POST_PROCESS_RUN_STATES = {
        StateCode.draft: {StateCode.prepare_default_run},
        StateCode.prepare_default_run: {StateCode.prepare_pp},
        StateCode.prepare_pp: {StateCode.final_eval},
        StateCode.final_eval: {StateCode.done, StateCode.failed},
        StateCode.done: set(),
        StateCode.failed: set()
    }
    return POST_PROCESS_RUN_STATES

def get_final_evaluation_run_states():
    FINAL_EVAL_RUN_STATES = {
        StateCode.draft: {StateCode.prepare_default_run},
        StateCode.prepare_default_run: {StateCode.prepare_pp},
        StateCode.prepare_pp: {StateCode.final_eval},
        StateCode.final_eval: {StateCode.done, StateCode.failed},
        StateCode.done: set(),
        StateCode.failed: set()
    }
    return FINAL_EVAL_RUN_STATES

RUN_TYPE_MAPPING = {
    AvailableRunTypes.base: get_base_run_states,
    AvailableRunTypes.fine_tune: get_fine_tune_states,
    AvailableRunTypes.post_process: get_post_process_run_states,
    # AvailableRunTypes.final_evaluation:  get_final_evaluation_run_states
}

def get_state_mappings(run_type):
    if run_type not in RUN_TYPE_MAPPING:
        raise ValueError(f'Run type "{run_type}" is not present in the RUN_TYPE_MAPPING')
    return RUN_TYPE_MAPPING[run_type](run_type)