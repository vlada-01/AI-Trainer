from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import sklearn.datasets as datasets
import numpy as np

from model_src.data.metas.meta import MetaTypes
from model_src.data.metas.utils import create_meta

from model_src.data.transforms import compose_transforms

from common.logger import get_logger

log = get_logger(__name__)

class SklearnDataBuilder():    
    def __init__(self, cfg, cfg_dataset_transforms):
        self.cfg = cfg
        self.cfg_dataset_transforms = cfg_dataset_transforms

        log.info('Loading raw data')
        X, y, meta = self.load_data(cfg)
        log.debug(f'Creating TabularMetaDta for meta: {meta}')
        self.meta = create_meta(MetaTypes.tabular, task=cfg.task, **meta)

        log.info('Splitting data')
        X_train, X_val, X_test, y_train, y_val, y_test = self.get_splits(X, y)

        log.info('Assembling dataset transformations')
        train_t, train_tt, val_t, val_tt, test_t, test_tt = self.assemble_transforms()

        log.info('Initializing Datasets')
        self.train_ds = SklearnDataset(X_train, y_train, transform=train_t, target_transform=train_tt)
        self.val_ds = SklearnDataset(X_val, y_val, transform=val_t, target_transform=val_tt)
        self.test_ds = SklearnDataset(X_test, y_test, transform=test_t, target_transform=test_tt)  if X_test is not None else None
        
        # tr_x, tr_y = self.train_ds[0]
        # val_x, val_y = self.val_ds[0]
        # test_x, test_y = self.test_ds[0] if self.test_ds is not None else (None, None)
        # train_sample = {
        #     'X': tr_x,
        #     'y': tr_y
        # }
        # val_sample = {
        #     'X': val_x,
        #     'y': val_y
        # }
        # test_sample = {
        #     'X': test_x,
        #     'y': test_y
        # }
        # self.meta.add_sample_info(train_sample, val_sample, test_sample)

    def get_train(self):
        return self.train_ds
    
    def get_val(self):
        return self.val_ds
    
    def get_test(self):
        return self.test_ds
    
    def get_meta(self):
        return self.meta

    def load_data(self, cfg):
        dataset_fn = cfg.dataset_fn
        if not hasattr(datasets, dataset_fn):
            raise ValueError(
                f"Dataset function '{dataset_fn}' not found in {datasets.__name__}"
            )
        fn = getattr(datasets, dataset_fn)
        if not callable(fn):
            raise TypeError(f"{dataset_fn} is not callable")
        data = fn(as_frame=True)
    
        X = data.data.to_numpy(dtype=np.float32)
        if  cfg.task == 'classification':
            y = data.target.to_numpy(dtype=np.long)
        else:
            y = data.target.to_numpy(dtype=np.float32)
        meta = self.extract_metadata(data)
        return X, y, meta
    
    def extract_metadata(self, data):
        descr = data.data.describe()
        mean = descr.loc["mean"].to_numpy()
        std  = descr.loc["std"].to_numpy()
        min = descr.loc["min"].to_numpy()
        max = descr.loc["max"].to_numpy()
        unique_targets = np.unique(data.target.to_numpy()).size
        rows_features_size = data.data.to_numpy().shape
        size = rows_features_size[0]
        features = rows_features_size[1]
        return {
            "size": size,
            "features": features,
            "unique_targets": unique_targets,
            "mean": mean,
            "std": std,
            "min": min,
            "max": max,
        }
    
    def get_splits(self, X, y):
        test = self.cfg.test_size
        stratify = y if (self.cfg.stratify and self.meta.get_task() == 'classification') else None
        X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test, stratify=stratify, shuffle=True) if test is not None else (X, None, y, None)

        val = self.cfg.val_size
        stratify = y_tmp if (self.cfg.stratify and self.meta.get_task() == 'classification') else None
        X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val, stratify=stratify, shuffle=True)
        return X_train, X_val, X_test, y_train, y_val, y_test

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

class SklearnDataset(Dataset):
    def __init__(self, X, target, transform, target_transform):
        self.X = X
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        x, y = self.X[idx], self.target[idx]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)
        
        return x, y
    
    def __len__(self):
        return len(self.X)

