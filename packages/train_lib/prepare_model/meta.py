def create_meta(cfg):
    return ModelMeta(cfg)

class ModelMeta:
    def __init__(self, cfg):
        self.specs_mapping = cfg.specs_mapping if cfg is not None else None

    def specs_mapper(self, key):
        if self.specs_mapping is not None and key in self.specs_mapping:
            return self.specs_mapping[key]
        return key
    
    def to_dict(self):
        return {}