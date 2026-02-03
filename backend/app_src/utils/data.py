from datasets import load_dataset_builder

from model_src.data.dataset_builders.builder import build_data

def get_dataset_info(cfg):
    name = cfg.name
    builder = load_dataset_builder(cfg.id, cfg.name)
    return builder.info

# TODO: If dataset does not provide validation set, need to manually split train dataset
# TODO: need to add more tfs
def prepare_dataset(cfg):
    train, val, _, meta =  build_data(cfg)
    return {
        'train': train,
        'val': val,
        'meta': meta,
        'cfg': cfg
    }