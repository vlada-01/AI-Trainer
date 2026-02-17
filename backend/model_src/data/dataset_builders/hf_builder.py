import os
from datasets import load_dataset, DatasetDict

from torch.utils.data import Dataset
from pprint import pformat

from model_src.data.metas.utils import create_meta, update_meta
from model_src.data.transforms import assemble_transforms

from common.logger import get_logger

log = get_logger(__name__)

seed = int(os.getenv("SEED", "42"))

MAX_LEN = 256

class HuggingFaceBuilder():
    def __init__(self, cfg, cfg_dataset_transforms):
        self.cfg = cfg
        self.cfg_dataset_transforms = cfg_dataset_transforms

        log.info('Loading raw data')
        raw = self.load_raw()

        log.info(f'Initializing the {cfg.meta_type.value} type')
        # TODO: there is no support for the tabular hf datasets (e.g. set_sizes, preprocess_raw)
        self.meta = create_meta(cfg.meta_type)

        log.info('Initializing preprocessing raw data')
        raw = self.preprocess_data(raw)

        log.info('Initializing split of raw data')
        raw_train, raw_val, raw_test = self.split_raw(raw)

        upd_dict = {
            'set_max_len': MAX_LEN,
            'prepare_textual_params': (raw_train)
        }
        log.debug('Updating meta with dict:\n%s', pformat({k: type(v).__name__ for k, v in upd_dict.items()}))
        update_meta(self.meta, upd_dict)

        log.info('Assembling dataset transformations')
        train_t, train_tt, val_t, val_tt, test_t, test_tt = assemble_transforms(self.cfg_dataset_transforms, self.meta)
        
        log.info('Initializing Datasets')
        self.train_ds = HfDataset(raw_train, train_t, train_tt)
        self.val_ds = HfDataset(raw_val, val_t, val_tt)
        self.test_ds = HfDataset(raw_test, test_t, test_tt)
        
        upd_dict = {
            'set_task': cfg.task,
            'set_input_keys': self.train_ds[0],
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
    
    def load_raw(self):
        kwargs = self.cfg.load_ds_args
        kwargs = {
            'path': self.cfg.id,
            'name': self.cfg.name,
            **kwargs
        }
        return load_dataset(**kwargs)
    
    def preprocess_data(self, raw):
        def normalize_inputs(batch):
            x_keys = self.cfg.x_keys
            y_keys = self.cfg.y_keys

            # return {
            #     'x': {k: batch[k] for k in x_keys},
            #     'y': {k: batch[k] for k in y_keys},
            # }
            # TODO: need to extend functionality to be able to preprocess x dict
            return {
                'x': batch[x_keys[0]],
                'y': batch[y_keys[0]]
            }

        log.info('Normalizing raw data to (x, y)')
        ds = raw.map(normalize_inputs, batched=True, num_proc=4)
        ds = ds.select_columns(['x', 'y'])

        ds = self.meta.preprocess_raw(ds)

        return ds

    def split_raw(self, raw):
        if not isinstance(raw, DatasetDict):
            raw = DatasetDict({'data': raw})

        train = raw.get(self.cfg.splits.train)
        val = raw.get(self.cfg.splits.val)
        test = raw.get(self.cfg.splits.test)

        if val is None:
            log.info('Validation set is not configured, falling back to ratios split')
            tmp_split = train.train_test_split(test_size=self.cfg.ratios.val)
            train = tmp_split['train']
            val = tmp_split['test']

        if test is None:
            log.info('Test set is not configured, falling back to ratios split')
            tmp_split = train.train_test_split(test_size=self.cfg.ratios.test, seed=seed)
            train = tmp_split['train']
            test = tmp_split['test']

        log.info(f'Train size: {len(train)}, Validation size: {len(val)}, Test size: {len(test)}')
        return train, val, test
    

class HfDataset(Dataset):
    def __init__(self, raw_ds, transform, target_transform):
        self.ds = raw_ds
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        raw_dict = self.ds[i]
        X, target = raw_dict['x'], raw_dict['y']
        if self.transform is not None:
            X = self.transform(X)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        if isinstance(X, dict):
            return {'X': X, 'y': target}, i
        
        return {'X': {'X':X}, 'y': target}, i
    
    def __len__(self):
        return len(self.ds)