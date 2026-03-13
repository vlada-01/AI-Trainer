from abc import ABC, abstractmethod
from enum import Enum

class MetaTypes(str, Enum):
    # tabular = 'tabular'
    image = 'image'
    textual = 'textual'

class DataMeta(ABC):
    def __init__(self, modality, specs_mapping):
        self.modality = modality
        self.specs_mapping = specs_mapping

    def specs_mapper(self, key):
        if self.specs_mapping is not None and key in self.specs_mapping:
            return self.specs_mapping[key]
        return key

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
    def get_output_unique_values(self, key):
        pass

    @abstractmethod
    def to_dict(self):
        pass