from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader

from model_src.data.metas.utils import create_meta
from model_src.data.mapper import prepare_mapper
from model_src.data.transforms import compose_transforms

from common.logger import get_logger

log = get_logger(__name__)

# {
#    "data_config": {
#     "dataset_provider": "hugging face",
#     "id": "fashion_mnist",
#     "name": null,
#     "task": "classification",
#     "meta_type": "image",
#     "train_split": "train",
#     "val_split": "test",
#     "test_split": null,
#     "load_ds_args": {},
#     "mapper": {
#       "name": "simple",
#       "x_mapping": "image",
#       "y_mapping": "label"
#     }
#   },
#   "dataset_transforms": {
#     "train": {
#       "transform": [
#         {
#           "name": "img_to_tensor",
#           "value": true
#         }
#       ],
#       "target_transform": [
#         {
#           "name": "to_tensor",
#           "value": true
#         }
#       ]
#     },
#     "valid": {
#       "transform": [
#         {
#           "name": "img_to_tensor",
#           "value": true
#         }
#       ],
#       "target_transform": [
#         {
#           "name": "to_tensor",
#           "value": true
#         }
#       ]
#     },
#     "test": null
#   },
#   "batch_size": 64,
#   "shuffle": true
# }

class HuggingFaceBuilder():
    def __init__(self, cfg, cfg_dataset_transforms):
        self.cfg = cfg
        self.cfg_dataset_transforms = cfg_dataset_transforms

        log.info('Initializing load of raw data')
        raw_train, raw_val, raw_test = self.load_data()
        
        log.info('Preparing mapper')
        mapper = prepare_mapper(cfg.mapper)

        # TODO: later when different classification image problems or text appears, this needs to be updated
        
        kwargs = {
            'task': cfg.task
        }
        self.meta = create_meta(cfg.meta_type, **kwargs)

        log.info('Initialize composing transforms')
        train_t, train_tt, val_t, val_tt, test_t, test_tt = self.assemble_transforms()
        
        log.info('Creating Datasets')
        self.train_ds = HfDataset(raw_train, mapper, train_t, train_tt)
        self.val_ds = HfDataset(raw_val, mapper, val_t, val_tt)
        self.test_ds = HfDataset(raw_test, mapper, test_t, test_tt) if raw_test is not None else None
        
        X, _ = self.train_ds[0]
        log.info('Searching for the unique targets')
        num_classes = self.find_uniques()
        log.info(f'Number of classes: {num_classes}')
        self.meta.input = X.size()
        self.meta.output = num_classes
        

    def get_train(self):
        return self.train_ds
    
    def get_val(self):
        return self.val_ds
    
    def get_test(self):
        return self.test_ds
    
    def get_meta(self):
        return self.meta

    def load_data(self):
        kwargs = self.cfg.load_ds_args
        name = self.cfg.name
        id = self.cfg.id
        if name is None:
            raw_train = load_dataset(id, split=self.cfg.train_split, **kwargs)
            raw_val = load_dataset(id, split=self.cfg.val_split, **kwargs)
            raw_test = load_dataset(id, split=self.cfg.test_split, **kwargs) if self.cfg.test_split is not None else None
        else:
            raw_train = load_dataset(id, name, split=self.cfg.train_split, **kwargs)
            raw_val = load_dataset(id, name, split=self.cfg.val_split, **kwargs)
            raw_test = load_dataset(id, name, split=self.cfg.test_split, **kwargs) if self.cfg.test_split is not None else None
        return raw_train, raw_val, raw_test
    
    def assemble_transforms(self):
        train_t, train_tt = None, None
        val_t, val_tt = None, None
        test_t, test_tt = None, None

        cfg = self.cfg_dataset_transforms
        if cfg is not None:
            t_cfg = cfg.train.transform
            tt_cfg = cfg.train.target_transform
            log.info('Composing train transforms')
            train_t = compose_transforms(t_cfg, self.meta)
            log.info('Composing train target transforms')
            train_tt = compose_transforms(tt_cfg, self.meta)

            t_cfg = cfg.valid.transform 
            tt_cfg = cfg.valid.target_transform
            log.info('Composing val transforms') 
            val_t = compose_transforms(t_cfg, self.meta)
            log.info('Composing val target transforms')
            val_tt = compose_transforms(tt_cfg, self.meta)

            t_cfg = cfg.test.transform if cfg.test is not None else None
            tt_cfg = cfg.test.target_transform if cfg.test is not None else None
            log.info('Composing test transforms') 
            test_t = compose_transforms(t_cfg, self.meta)
            log.info('Composing test target transforms') 
            test_tt = compose_transforms(tt_cfg, self.meta)
        return train_t, train_tt, val_t, val_tt, test_t, test_tt
    
    def find_uniques(self):
        loader = DataLoader(self.train_ds, batch_size=512, shuffle=False, num_workers=0)

        seen = set()
        for batch in loader:
            _, y = batch
            if hasattr(y, "tolist"):
                seen.update(y.tolist())
            else:
                seen.update(list(y))
        num_classes = len(seen)
        return num_classes

class HfDataset(Dataset):
    def __init__(self, raw_ds, mapper, transform, target_transform):
        self.ds = raw_ds
        self.mapper = mapper
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        raw_dict = self.ds[i]
        X, target = self.mapper(raw_dict)
        if self.transform is not None:
            X = self.transform(X)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return X, target
    
    def __len__(self):
        return len(self.ds)