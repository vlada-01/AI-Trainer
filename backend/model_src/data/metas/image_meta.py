from torch.utils.data import DataLoader
from model_src.data.metas.meta import MetaData, MetaTypes
import torch

from common.logger import get_logger

log = get_logger(__name__)

class ImageMetaData(MetaData):
    def __init__(self):
        super().__init__(MetaTypes.image)
        self.task = None
        self.input_size = None
        self.output_size = None

    def update(self, upd_dict):
        for k, v in upd_dict.items():
            if hasattr(self, k):
                if callable(getattr(self, k)):
                    log.debug(f'Calling ImageMetaData attr "{k}"')
                    fn = getattr(self, k)
                    if isinstance(v, dict):
                        fn(**v)
                    elif isinstance(v, (list, tuple)):
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
    
    def get_necessary_sizes(self):
        return {
            'input_size': self.input_size,
            'output_size': self.output_size
        }
    
    def get_task(self):
        return self.task
    
    def get_unique_targets(self):
        return int(self.output_size[0])
    
    def to_dict(self):
        return {
            'modality': self.modality,
            'input_size': list(self.input_size),
            'output_size': list(self.output_size),
            'task': self.task,
        }
    
    # ---------------------- Resolvers ----------------------
    
    def resolve_img_to_tensor(self):
        return {}
    
    def resolve_to_tensor(self):
        return {}
    
    # ---------------------- Internal -----------------------
    
    # TODO: need to be careful what is the output.size() - 1 num, list, list[list]
    def set_sizes(self, train_ds):
        log.debug('Updating ImageMetaData sizes')
        if self.task == 'regression':
            X, y = train_ds[0]
            self.input_size = X.size()
            self.output_size = y.size()
            return

        loader = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=0)

        seen = set()
        for batch in loader:
            _, y = batch
            if hasattr(y, "tolist"):
                seen.update(y.tolist())
            else:
                seen.update(list(y))

        num_classes = len(seen)

        log.debug(f'Number of classes: {num_classes}')
        
        X, _ = train_ds[0]
        self.input_size = X.size()
        self.output_size = torch.Size([num_classes])

    def set_task(self, task):
        self.task = task