from model_src.data.metas.meta import MetaData, MetaTypes

class TabularMetaData(MetaData):
    def __init__(self, size, features, unique_targets, mean, std, min, max, task, extras=None, tr_sample=None, val_sample=None, test_sample=None):
        super().__init__(MetaTypes.tabular, extras, tr_sample, val_sample, test_sample)
        self.size = size
        self.features = features
        self.unique_targets = unique_targets
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max
        self.task = task

    def resolve(self, name):
        fn = getattr(self, f'resolve_{name.value}', None)
        if fn is None:
            raise NameError(f'Name resolve_{name.value} is not supported in {type(self)}')
        return name, fn()
    
    # def add_sample_info(self, tr, val, test):
    #     self.train_sample = tr
    #     self.val_sample = val
    #     self.test_sample = test
    
    def get_sample_sizes(self):
        return {
            'sample': self.get_sample_size(),
            # 'train_sample': self.get_sample_size(self.train_sample),
            # 'val_sample': self.get_sample_size(self.val_sample),
            # 'test_sample': self.get_sample_size(self.test_sample)
        }
    
    def get_task(self):
        return self.task
    
    def get_unique_targets(self):
        return self.unique_targets
    
    def to_dict(self):
        return {
            'modality': self.modality,
            'extras': self.extras,
            'size': self.size,
            'features': self.features,
            'unique_targets': self.unique_targets,
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'min': self.min.tolist(),
            'max': self.max.tolist(),
            'task': self.task,
        }
    
    def get_sample_size(self):
        # if sample is None:
        #     return None 
        
        return {
            'input_size': self.features,
            'output_size': self.unique_targets if self.task == 'classification' else 1
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