from packages.train_lib.prepare_data.metas.meta import DataMeta
from packages.train_lib.prepare_train.meta import TrainMeta
from packages.train_lib.prepare_model.meta import ModelMeta

class Meta:
    def __init__(self, specs_cfg):
        self.specs = specs_cfg

        self.data_meta: DataMeta = None
        self.model_meta: ModelMeta = None
        self.train_meta: TrainMeta = None
 
    def set(self, attr, val):
        if hasattr(self, attr):
            setattr(self, attr, val)
        raise ValueError(f'Meta does not have attr: {attr}')

    def get_data_meta(self):
        return self.data_meta
    
    def get_model_meta(self):
        return self.model_meta
    
    def get_train_meta(self):
        return self.train_meta
    
    def get_specs(self):
        return self.specs
    
    def to_dict(self):
        return {
            'specs': self.specs,
            'data_meta': self.data_meta.to_dict(),
            'model_meta': self.model_meta.to_dict(),
            'train_meta': self.train_meta.to_dict(),
        }
            
