from datasets import load_dataset_builder

def get_dataset_info(cfg):
    builder = load_dataset_builder(cfg.id, cfg.name)
    return builder.info