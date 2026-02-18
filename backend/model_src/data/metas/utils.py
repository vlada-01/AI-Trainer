
from model_src.data.metas.meta import MetaTypes
from model_src.data.metas.tabular_meta import TabularMetaData
from model_src.data.metas.image_meta import ImageMetaData
from model_src.data.metas.textual_meta import TextualMetaData

from common.logger import get_logger

log = get_logger(__name__)

META_DATA_REGISTRY_MAP = {
    MetaTypes.tabular: TabularMetaData,
    MetaTypes.image: ImageMetaData,
    MetaTypes.textual: TextualMetaData
}

def create_meta(meta_type, preconfigured_dict):
    if meta_type not in META_DATA_REGISTRY_MAP:
        raise ValueError(f'{meta_type} not supported in Metas')
    meta = META_DATA_REGISTRY_MAP[meta_type]()
    for k, v in preconfigured_dict.items():
        if hasattr(meta, k):
            log.debug(f'Initializing meta field "{k}" with preconfigured value')
            setattr(meta, k, v)
        else:
            raise ValueError(f'Metatype: {meta_type} does not support field {k}')
    return meta

def update_meta(meta, upd_dict, preconfigured=False):
    if not preconfigured:
        meta.update(upd_dict)
