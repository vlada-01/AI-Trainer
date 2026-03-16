from enum import Enum
from abc import ABC, abstractmethod

class AvailablePostProcessors(str, Enum):
    #for classification
    calibration = 'calibration'
    global_threshold = 'global_threshold'

class PostProcessor(ABC):
    def __init__(self, name, in_key, out_key, fallback_key, trainable=False):
        self.name = name
        self.in_key = in_key
        self.out_key = out_key
        self.trainable = trainable
        self.fallback_key = fallback_key

    def resolve(self, state):
        return state

    @abstractmethod
    def process(self, state, return_details=False):
        pass

    def is_trainable(self):
        return self.trainable
    
    @abstractmethod
    def train(self, model, val, device):
        pass

    def get_name(self):
        return self.name

    def get_in_key(self):
        return self.in_key

    def get_out_key(self):
        return self.out_key