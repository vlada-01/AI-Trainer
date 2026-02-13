from datasets import load_dataset_builder

from model_src.data.dataset_builders.builder import build_data

from common.logger import get_logger

log = get_logger(__name__)

def get_dataset_info(cfg):
    builder = load_dataset_builder(cfg.id, cfg.name)
    return builder.info

def atomic_prepare_dataset(cfg):
    log.info('Initializing prepare dataset process')
    train, val, _, meta =  build_data(cfg)
    log.info('Prepare dataset is successfullty finished')
    ctx_dict = {
        'train': train,
        'val': val,
        'meta': meta,
        'cached_dl_cfg': cfg
    }
    result = {
        'sample_size': meta.get_necessary_sizes()
    }
    return result, ctx_dict