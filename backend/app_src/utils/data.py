from datasets import load_dataset_builder

from model_src.data.dataset_builders.builder import build_data

from common.logger import get_logger

log = get_logger(__name__)

def get_dataset_info(cfg):
    builder = load_dataset_builder(cfg.id, cfg.name)
    return builder.info

# TODO: If dataset does not provide validation set, need to manually split train dataset
def prepare_dataset(cfg):
    log.info('Initializing prepare dataset process')
    train, val, _, meta =  build_data(cfg)
    log.info('Prepare dataset is successfullty finished')
    return {
        'train': train,
        'val': val,
        'meta': meta,
        'cfg': cfg
    }