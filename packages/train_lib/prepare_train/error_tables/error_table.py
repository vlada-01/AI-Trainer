from abc import ABC, abstractmethod
import pandas as pd

class ErrorTable(ABC):
    def __init__(self):
        self.df = pd.DataFrame()

    @abstractmethod
    def set_states(self, meta, key):
        pass

    @abstractmethod
    def restart_error_table(self):
        pass

    @abstractmethod
    def restart(self):
        pass
    
    @abstractmethod
    def update(self, ids, h_outs, targets):
        pass

    @abstractmethod
    def collect_error_table(self):
        pass

    def collect_extras(self):
        return {}