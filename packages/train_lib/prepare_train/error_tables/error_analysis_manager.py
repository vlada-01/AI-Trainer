from packages.train_lib.tasks import AvailableTasks

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

    def restart(self):
        for v in self.error_analysis.values():
            v.restart()

    def restart_error_tables(self):
        for v in self.error_analysis.values():
            v.restart_error_table()

    def update(self, ids, h_outs, targets):
        for k, v in self.error_analysis.items():
            curr_h_outs = h_outs[k]
            curr_targets = targets[k]
            v.update(ids, curr_h_outs, curr_targets)

    # returns pandas df
    def collect_error_tables(self):
        results = {}
        for k, v in self.error_analysis.items():
            results[k] = v.collect_error_table()
        return results

    def collect_extras(self):
        results = {}
        for k, v in self.error_analysis.items():
            results[k] = v.collect_extras()
        return results


