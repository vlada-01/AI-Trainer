import os
from datasets import load_dataset, DatasetDict, Features

from torch.utils.data import Dataset
from pprint import pformat

from model_src.data.metas.utils import create_meta, update_meta
from model_src.data.transforms import assemble_transforms

from common.logger import get_logger

log = get_logger(__name__)

seed = int(os.getenv("SEED", "42"))

MAX_LEN = 256

class HuggingFaceBuilder():
    def __init__(self, cfg, cfg_dataset_transforms, preconfigured_meta=None):
        self.cfg = cfg
        self.cfg_dataset_transforms = cfg_dataset_transforms

        log.info('Loading raw data')
        raw = self.load_raw()

        log.info(f'Initializing the {cfg.meta_type.value} type')
        # TODO: there is no support for the tabular hf datasets (e.g. set_sizes, preprocess_raw)
        self.meta = create_meta(cfg.meta_type, preconfigured_meta)

        log.info('Initializing preprocessing raw data')
        raw = self.preprocess_data(raw)

        log.info('Initializing split of raw data')
        raw_train, raw_val, raw_test = self.split_raw(raw)

        if preconfigured_meta is None:
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
        
        if preconfigured_meta is None:
            upd_dict = {
                'set_tasks': cfg.tasks,
                'set_input_keys': self.train_ds[0],
                'set_input_sizes': self.train_ds[0],
                'set_output_sizes': self.train_ds[0],
                'set_output_unique_values': self.train_ds,
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
        def get_by_path(batch, key):
            cur = batch
            for k in key.split("."):
                if isinstance(cur, list):
                    cur = [item[k] for item in cur]
                else:
                    cur = cur[k]
            # try:
            #     print('return: ', cur, len(cur))
            # except:
            #     print('return: ', cur)
            return cur

        def build_xy_features(raw_features):
            x_keys = self.cfg.x_keys
            y_keys = self.cfg.y_keys

            x = {k.split('.')[-1]: get_by_path(raw_features, k) for k in x_keys}
            y = {k.split('.')[-1]: get_by_path(raw_features, k) for k in y_keys}
            return Features({'x': x, 'y': y})
        
        def normalize_inputs(batch):
            x_keys = self.cfg.x_keys
            y_keys = self.cfg.y_keys
            xs_dict = {k.split('.')[-1]: get_by_path(batch, k) for k in x_keys}
            # xs_dict = {'image': batch['image']}
            ys_dict = {k.split('.')[-1]: get_by_path(batch, k) for k in y_keys}
            
            def columns_to_rows(cols: dict):
                keys = list(cols.keys())
                n = len(next(iter(cols.values())))
                return [
                    {k: cols[k][i] for k in keys}
                    for i in range(n)
                ]
            
            final = {
                'y': columns_to_rows(ys_dict),
                'x': columns_to_rows(xs_dict),
                # 'x': columns_to_rows(xs_dict)
                # 'x': {'image': batch['image'], 'image_id': batch['image_id'], 'width': batch['width'], 'height': batch['height']}
            }
            # print(final['x'][0])
            return final

        log.info('Normalizing raw data to (x, y)')
        split_keys = list(raw.keys())
        raw_features = raw[split_keys[0]].features
        xy_features = build_xy_features(raw_features)
        # print(xy_features)
        # print(raw['train']['image'][0])
        ds = raw.map(normalize_inputs, batched=True, num_proc=4)
        ds = ds.select_columns(['x', 'y'])
        ds = ds.cast(xy_features)
        # print(ds['train']['x'][0])
        log.info('Mapp done')

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
            tmp = {}
            for k, v in X.items():
                v_tfd = self.transform(v)
                if isinstance(v_tfd, dict):
                    tmp = {**tmp, **v_tfd}
                else:
                    tmp[k] = v_tfd
            X = tmp
        if self.target_transform is not None:
            tmp = {}
            for k, v in target.items():
                v_tfd = self.target_transform(v)
                if isinstance(v_tfd, dict):
                    tmp = {**tmp, **v_tfd}
                else:
                    tmp[k] = v_tfd
            target = tmp  
        return {'X': X, 'y': target}, i
        
    def __len__(self):
        return len(self.ds)