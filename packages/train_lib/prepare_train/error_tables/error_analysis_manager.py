from abc import ABC, abstractmethod

from packages.train_lib.meta import AvailableTasks

from packages.train_lib.prepare_train.error_tables.classification.error_table import ClassificationErrorTable
from packages.train_lib.prepare_train.error_tables.regression.error_table import RegressionErrorTable

from packages.logger.logger import get_logger

log = get_logger(__name__)

ERROR_TABLE_REGISTRY_MAP = {
    AvailableTasks.classification: ClassificationErrorTable,
    AvailableTasks.regression: RegressionErrorTable
}

def prepare_error_analysis(meta):
    log.info('Initializing ErrorAnalysisManager')
    error_analysis_dict = {}
    for k, specs in meta.get_specs():
        task = specs.task
        log.info(f'Initializing ErrorTable for the task: {task}')
        if task not in ERROR_TABLE_REGISTRY_MAP:
            raise ValueError(f'Error table does not exist for the task: {task.value}')
        error_table = ERROR_TABLE_REGISTRY_MAP[task]()
        error_table.set_states(meta, k)
        error_analysis_dict[k] = error_table

    error_analysis = ErrorAnalysisManager(error_analysis_dict)
    log.info('ErrorAnalysisManager successfully prepared')
    return error_analysis

class ErrorAnalysisManager:
    def __init__(self, error_analysis_dict):
        self.error_analysis = error_analysis_dict

    def restart_error_tables(self):
        for v in self.error_analysis.values():
            v.restart()

    def update_error_tables(self, ids, h_outs, targets):
        for k, v in self.error_analysis.items():
            curr_h_outs = h_outs[k]
            curr_targets = targets[k]
            # TODO: update this crap inside the specific error table
            v.update(ids, curr_h_outs, curr_targets)

    def test_update_error_analysis(self, ids, preds, targets):
        for k, v in self.items():
            curr_preds = preds[k]
            curr_targets = targets[k]
            # TODO: update this crap inside the specific error table
            v.test_update(ids, curr_preds, curr_targets)

    def get_results(self):
        results = {}
        for k, v in self.error_analysis.items():
            results[k] = v.get_results()
        return results

class ErrorTable(ABC):
    @abstractmethod
    def set_states(self, meta, key):
        pass

    @abstractmethod
    def restart(self):
        pass
    
    @abstractmethod
    def update(self, ids, h_outs, targets):
        pass

    @abstractmethod
    def test_update(self, ids, h_outs, targets):
        pass

    @abstractmethod
    def get_results(self):
        pass
