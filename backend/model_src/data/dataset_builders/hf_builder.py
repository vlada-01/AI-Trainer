from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from pprint import pformat

from model_src.data.metas.utils import create_meta, update_meta
from model_src.data.mapper import prepare_mapper
from model_src.data.transforms import assemble_transforms

from common.logger import get_logger

log = get_logger(__name__)

MAX_LEN = 256

class HuggingFaceBuilder():
    def __init__(self, cfg, cfg_dataset_transforms):
        self.cfg = cfg
        self.cfg_dataset_transforms = cfg_dataset_transforms

        log.info('Initializing load of raw data')
        raw_train, raw_val, raw_test = self.load_data()

        log.info('Initializing preprocessing raw data')
        mapper, (raw_train, raw_val, raw_test) = self.preprocess_data([raw_train, raw_val, raw_test])
        
        log.info(f'Initializing the {cfg.meta_type.value} type')
        # TODO: there is no support for the tabular hf datasets (e.g. set_sizes)
        self.meta = create_meta(cfg.meta_type)

        upd_dict = {
            'set_max_len': MAX_LEN,
            'prepare_textual_params': (raw_train, mapper)
        }
        log.debug('Updating meta with dict:\n%s', pformat({k: type(v).__name__ for k, v in upd_dict.items()}))
        update_meta(self.meta, upd_dict)

        log.info('Assembling dataset transformations')
        train_t, train_tt, val_t, val_tt, test_t, test_tt = assemble_transforms(self.cfg_dataset_transforms, self.meta)
        
        log.info('Initializing Datasets')
        self.train_ds = HfDataset(raw_train, mapper, train_t, train_tt)
        self.val_ds = HfDataset(raw_val, mapper, val_t, val_tt)
        self.test_ds = HfDataset(raw_test, mapper, test_t, test_tt) if raw_test is not None else None
        
        upd_dict = {
            'set_task': cfg.task,
            'set_sizes': self.train_ds,
        }
        log.debug('Updating meta with dict:\n%s', pformat({k: type(v).__name__ for k, v in upd_dict.items()}))
        update_meta(self.meta, upd_dict)

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
        kwargs = {
            'path': self.cfg.id,
            'name': self.cfg.name,
            **kwargs
        }
        splits = [
            self.cfg.train_split,
            self.cfg.val_split,
            self.cfg.test_split
        ]
        raws = []
        for split in splits:
            kwargs = {
                **kwargs,
                'split': split
            }
            if split is None:
                raws.append(None)
            else:
                raws.append(load_dataset(**kwargs))
        return tuple(raws)
    
    # TODO: refactor later for any case to return standardized dictionary, easier to control
    def preprocess_data(self, raws):
        if self.cfg.meta_type.value == 'textual':
            log.info('Preprocessing Textual datasets with "distilbert-base-uncased" tokenizer')
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
            
            def tokenize_fn(batch):
                texts = batch[self.cfg.mapper.x_mapping]
                return {
                    'tokens': [tokenizer.tokenize(t) for t in texts]
                }
            for i, raw in enumerate(raws):
                if raw is not None: 
                    raws[i] = raw.map(tokenize_fn, batched=True, num_proc=4)
            log.info('Preparing mapper')
            self.cfg.mapper.x_mapping = 'tokens'
            mapper = prepare_mapper(self.cfg.mapper)
            return mapper, tuple(raws)
        log.info('Preparing mapper')
        mapper = prepare_mapper(self.cfg.mapper)
        return mapper, tuple(raws)

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