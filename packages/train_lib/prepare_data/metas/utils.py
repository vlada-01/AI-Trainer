
from .meta import MetaTypes
# from model_src.data.metas.tabular_meta import TabularMetaData
from .image_meta import ImageDataMeta
from .textual_meta import TextualDataMeta

from packages.logger.logger import get_logger

log = get_logger(__name__)

META_DATA_REGISTRY_MAP = {
    # MetaTypes.tabular: TabularMetaData,
    MetaTypes.image: ImageDataMeta,
    MetaTypes.textual: TextualDataMeta
}

def create_meta(cfg):
    log.info(f'Initializing the DataMeta of type: {cfg.meta_type.value}')
    meta_type = cfg.meta_type
    specs_mapping = cfg.specs_mapping
    if meta_type not in META_DATA_REGISTRY_MAP:
        raise ValueError(f'{meta_type} not supported in DataMetas')
    meta = META_DATA_REGISTRY_MAP[meta_type](specs_mapping)
    log.info('DataMeta successfully prepared')
    return meta

def update_meta(meta, upd_dict):
    meta.update(upd_dict)
