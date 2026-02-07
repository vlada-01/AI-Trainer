from model_src.data.metas.meta import MetaData, MetaTypes

from common.logger import get_logger

log = get_logger(__name__)

class TabularMetaData(MetaData):
    def __init__(self):
        super().__init__(MetaTypes.tabular)
        self.num_features = None
        self.unique_targets = None
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.task = None

    def update(self, upd_dict):
        for k, v in upd_dict.items():
            if hasattr(self, k):
                if callable(getattr(self, k)):
                    log.debug(f'Calling TabularMetaData attr "{k}"')
                    fn = getattr(self, k)
                    if isinstance(v, dict):
                        fn(**v)
                    elif isinstance(v, (list, tuple)):
                        fn(*v)
                    else:
                        fn(v)
                else:
                    raise ValueError(f'TabularMetaData does not have callable {k}')
            else:
                log.warning(f'TabularMetaData does not have attr {k}')

    def resolve(self, name):
        fn = getattr(self, f'resolve_{name.value}', None)
        if fn is None:
            raise NameError(f'Name resolve_{name.value} is not supported in {type(self)}')
        return name, fn()
    
    def get_sample_size(self):
        return {
            'input_size': self.num_features,
            'output_size': self.unique_targets if self.task == 'classification' else 1
        }
    
    def get_task(self):
        return self.task
    
    def get_unique_targets(self):
        return self.unique_targets
    
    # TODO: when adding support for the hf tabular, this needs to be updated self.mean.tolist(),
    def to_dict(self):
        return {
            'modality': self.modality,
            'num_features': self.num_features,
            'unique_targets': self.unique_targets,
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'min': self.min.tolist(),
            'max': self.max.tolist(),
            'task': self.task,
        }
    
    # ---------------------- Resolvers ----------------------

    def resolve_normalize_mean_std(self):
        params = {}
        params['mean'] = self.mean
        params['std'] = self.std
        return params
    
    def resolve_normalize_min_max(self):
        params = {}
        params['min'] = self.min
        params['max'] = self.max
        return params
    
    def resolve_to_tensor(self):
        return {}
    
    # ---------------------- Internal -----------------------

    def set_num_features(self, num_features):
        self.num_features = num_features
    
    def set_unique_targets(self, unique_targets):
        self.unique_targets = unique_targets

    def set_mean(self, mean):
        self.mean = mean

    def set_std(self, std):
        self.std = std

    def set_min(self, min):
        self.min = min

    def set_max(self, max):
        self.max = max

    def set_task(self, task):
        self.task = task