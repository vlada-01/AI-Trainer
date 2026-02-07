
from model_src.data.metas.meta import MetaTypes
from model_src.data.metas.tabular_meta import TabularMetaData
from model_src.data.metas.image_meta import ImageMetaData

META_DATA_REGISTRY_MAP = {
    MetaTypes.tabular: TabularMetaData,
    MetaTypes.image: ImageMetaData
}

def create_meta(meta_type):
    if meta_type not in META_DATA_REGISTRY_MAP:
        raise ValueError(f'{meta_type} not supported in Metas')
    return META_DATA_REGISTRY_MAP[meta_type]()

def update_meta(meta, upd_dict):
    meta.update(upd_dict)

# def try_rebuild_meta(meta_cfg):
#     if meta_cfg is None:
#         return None, False
#     meta_type = meta_cfg['modality']
#     meta = create_meta(meta_type)
#     for k, v in meta_cfg.items():
#         if hasattr(meta, k):
#             setattr(meta, k, v)
#         else:
#             raise ValueError(f'{meta_type} does not have field {k}')
#     return meta, True