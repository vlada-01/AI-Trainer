from torch.utils.data import DataLoader
from model_src.data.metas.meta import MetaData, MetaTypes
import torch
from pprint import pformat

from common.logger import get_logger

log = get_logger(__name__)

class ImageMetaData(MetaData):
    def __init__(self):
        super().__init__(MetaTypes.image)
        self.tasks = None
        self.input_sizes = None
        self.output_sizes = None
        self.output_unique_values = None
        self.input_keys = None

    def preprocess_raw(self, ds):
        return ds

    def update(self, upd_dict):
        for k, v in upd_dict.items():
            if hasattr(self, k):
                if callable(getattr(self, k)):
                    log.debug(f'Calling ImageMetaData attr "{k}"')
                    fn = getattr(self, k)
                    if isinstance(v, dict):
                        fn(**v)
                    elif isinstance(v, tuple):
                        fn(*v)
                    else:
                        fn(v)
                else:
                    raise ValueError(f'ImageMetaData does not have callable {k}')
            else:
                log.warning(f'ImageMetaData does not have attr {k}')

    def resolve(self, name):
        fn = getattr(self, f'resolve_{name.value}', None)
        if fn is None:
            raise NameError(f'Name resolve_{name.value} is not supported in {type(self)}')
        return name, fn()
    
    def get_input_keys(self):
        return self.input_keys
    
    def get_output_unique_values(self, key):
        return self.output_unique_values[key]

    def get_necessary_sizes(self):
        return {
            'input_sizes': self.input_sizes,
            'output_sizes': self.output_sizes,
            'output_unique_values': self.output_unique_values,
        }
    
    def get_tasks(self):
        return self.tasks
    
    def to_dict(self):
        return {
            'modality': self.modality,
            'input_sizes': self.input_sizes,
            'output_sizes': self.output_sizes,
            'output_unique_values': self.output_unique_values,
            'input_keys': self.input_keys,
            'tasks': self.tasks,
        }
    
    # ---------------------- Resolvers ----------------------
    
    def resolve_img_to_tensor(self):
        return {}
    
    def resolve_to_tensor(self):
        return {}
    
    # ---------------------- Internal -----------------------
    
    def set_input_sizes(self, sample, id):
        X = sample['X']
        self.input_sizes = {k: list(v.size()) for k, v in X.items()}
    
    def set_output_sizes(self, sample, id):
        y = sample['y']
        self.output_sizes = {k: list(v.size()) for k, v in y.items()}
    
    # TODO: update me according to new y: dict
    def set_output_unique_values(self, train_ds):
        log.info('Updating ImageMetaData sizes')

        loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=0)
        sample_dict, _ = train_ds[0]
        y_keys = sample_dict['y'].keys()
        seen = {k: set() for k in y_keys}
        for samples, _ in loader:
            y = samples['y']
            _ = [set.update(vals.tolist()) for set in seen.values() for vals in y.values()]
            # for k, v in y.items():
            #     if hasattr(v, "tolist"):
            #         seen
            #         seen.update(y.tolist())
            #     else:
            #         seen.update(list(y))
        seen = {k: len(v) for k, v in seen.items()}
        log.info(f'Number of uniqe values:\n%s', pformat({k: v for k, v in seen.items()}))
        
        self.output_unique_values = seen

    def set_tasks(self, tasks):
        self.tasks = tasks

    def set_input_keys(self, sample, id):
        X_dict = sample['X']
        self.input_keys = [k for k in X_dict.keys()]