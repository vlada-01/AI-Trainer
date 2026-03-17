from enum import Enum
from torch.utils.data import DataLoader

from .data_builder import HuggingFaceBuilder
from .metas import create_meta

from packages.logger import get_logger

log = get_logger(__name__)

class AvailableProviders(str, Enum):
    sklearn = 'sklearn'
    hf = 'hugging face'

BUILDER_MAP = {
    AvailableProviders.hf: HuggingFaceBuilder
}

def build_data(cfg):
    provider = cfg.data_config.dataset_provider
    log.info(f'Initializing data builder for the provider: {provider.value}')
    data_meta_cfg = cfg.data_meta_cfg
    # TODO: there is no support for the tabular datasets (e.g. set_sizes, preprocess_raw)
    data_meta = create_meta(data_meta_cfg)
    databuilder = BUILDER_MAP[provider](cfg.data_config, cfg.dataset_transforms, data_meta)
    log.info(f'Databuilder for provider {provider.value} is prepared successfully')
    
    batch_size = cfg.batch_size
    shuffle = cfg.shuffle
    log.info(f'Initializing DataLoaders for provider: {provider.value}')
    train_dl = DataLoader(databuilder.get_train(), shuffle=shuffle, batch_size=batch_size)
    val_dl = DataLoader(databuilder.get_val(), shuffle=shuffle, batch_size=batch_size)
    test_dl = DataLoader(databuilder.get_test(), shuffle=shuffle, batch_size=batch_size)
    log.info(f'DataLoaders for provider {provider.value} are prepared successfully')
    return train_dl, val_dl, test_dl, data_meta
