
from model_src.data.metas.meta import MetaTypes
from model_src.data.metas.tabular_meta import TabularMetaData
from model_src.data.metas.image_meta import ImageMetaData

META_DATA_REGISTRY_MAP = {
    MetaTypes.tabular: TabularMetaData,
    MetaTypes.image: ImageMetaData
}

def create_meta(meta_type, **kwargs):
    if meta_type not in META_DATA_REGISTRY_MAP:
        raise ValueError(f'{meta_type} not supported in Metas')
    return META_DATA_REGISTRY_MAP[meta_type](**kwargs)