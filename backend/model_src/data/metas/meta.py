from abc import ABC, abstractmethod
from enum import Enum

class MetaTypes(str, Enum):
    tabular = 'tabular'
    image = 'image'
    textual = 'textual'

class MetaData(ABC):
    def __init__(self, modality):
        self.modality = modality

    @abstractmethod
    def preprocess_raw(self, ds):
        pass

    @abstractmethod
    def update(self, upd_dict):
        pass

    @abstractmethod
    def resolve(self, name):
        pass

    @abstractmethod
    def get_input_keys(self, sample):
        pass

    @abstractmethod
    def get_necessary_sizes(self):
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