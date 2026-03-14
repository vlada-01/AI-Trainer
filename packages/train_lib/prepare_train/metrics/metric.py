from abc import ABC, abstractmethod
from enum import Enum

class AvailableMetrics(str, Enum):
    # for classifications
    accuracy = 'accuracy'
    precision = 'precision'
    recall = 'recall'
    f1_score = 'f1_score'
    # for regressions
    mse = 'mse'
    mae = 'mae'
    rmse = 'rmse'
    r2 = 'rs'
    # for textuals
    bleu = 'bleu'
    perplexity = 'perplexity'
    
    # add total exec time

class Metric(ABC):
    @abstractmethod
    def set_states(self, meta, key):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, preds, targets, post_processor):
        pass

    @abstractmethod
    def show(self):
        pass