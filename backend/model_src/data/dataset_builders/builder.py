from enum import Enum
from torch.utils.data import DataLoader, Dataset
from pprint import pformat

from model_src.data.dataset_builders.sklearn_builder import SklearnDataBuilder
from model_src.data.dataset_builders.hf_builder import HuggingFaceBuilder

from model_src.data.transforms import compose_transforms

from common.logger import get_logger

log = get_logger(__name__)

class AvailableProviders(str, Enum):
    sklearn = 'sklearn'
    hf = 'hugging face'

BUILDER_MAP = {
    AvailableProviders.sklearn: SklearnDataBuilder,
    AvailableProviders.hf: HuggingFaceBuilder
}

def build_data(cfg):
    provider = cfg.data_config.dataset_provider
    log.info(f'Initializing data builder for the provider: {provider.value}')
    databuilder = BUILDER_MAP[provider](cfg.data_config, cfg.dataset_transforms)
    log.info(f'Databuilder for provider {provider.value} is prepared successfully')
    
    batch_size = cfg.batch_size
    shuffle = cfg.shuffle
    log.info(f'Initializing DataLoaders for provider: {provider.value}')
    train_dl = DataLoader(databuilder.get_train(), shuffle=shuffle, batch_size=batch_size)
    val_dl = DataLoader(databuilder.get_val(), shuffle=shuffle, batch_size=batch_size)
    test_dl = DataLoader(databuilder.get_test(), shuffle=shuffle, batch_size=batch_size)
    log.info(f'DataLoaders for provider {provider.value} are prepared successfully')
    meta = databuilder.get_meta()
    return train_dl, val_dl, test_dl, meta

def update_train_data(train_dl, meta, new_ds_cfg):
    log.debug('Updating train dataloader with cfg:\n%s', pformat(new_ds_cfg.model_dump()))
    batch_size = train_dl.batch_size
    new_train_transform = new_ds_cfg.new_train_transform

    # CHECK_ME: need to be careful when adding transforms that change input size
    new_transform = compose_transforms(new_train_transform, meta)
    ds_wrapper = train_dl.dataset
    ds_wrapper.ds.transform = new_transform
    return DataLoader(ds_wrapper, batch_size=batch_size)

def update_dl_cfg(old_cfg, new_dl_cfg):
    log.debug('Updating train transforms with cfg:\n%s', pformat(new_dl_cfg.model_dump()))
    old_cfg.dataset_transforms.train.transform = new_dl_cfg.new_train_transform
    return old_cfg