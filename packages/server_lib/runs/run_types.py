from enum import Enum

class AvailableRunTypes(str, Enum):
    base = 'base'
    fine_tune = 'fine_tune'
    post_process = 'post_process'
    final_evaluation = 'final_evaluation'