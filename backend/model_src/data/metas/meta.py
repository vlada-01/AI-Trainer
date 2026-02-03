from abc import ABC, abstractmethod
from enum import Enum

class MetaTypes(str, Enum):
    tabular = 'tabular'
    image = 'image'

class MetaData(ABC):
    def __init__(self, modality, extras=None, train_sample=None, val_sample=None, test_sample=None):
        self.modality = modality
        self.extras = extras
        self.train_sample = train_sample
        self.val_sample = val_sample
        self.test_sample = test_sample

    @abstractmethod
    def resolve(self, name):
        pass

    # @abstractmethod
    # def add_sample_info(self):
    #     pass

    @abstractmethod
    def get_sample_sizes(self):
        pass

    @abstractmethod
    def get_task(self):
        pass

    @abstractmethod
    def get_unique_targets(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass