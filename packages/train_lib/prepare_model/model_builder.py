from .meta import create_meta
from .model import create_model

from packages.logger import get_logger

log = get_logger(__name__)

def build_model(meta, cfg):
    log.info('Initializing model builder')
    
    log.info('Initializing ModelMeta')
    model_meta_cfg = cfg.model_meta_cfg
    model_meta = create_meta(model_meta_cfg)
    
    specs_mapper = model_meta.specs_mapper
    pps_cfg = {specs_mapper(k): v for k, v in cfg.model_cfg.pps_cfg.items()}
    cfg.model_cfg.pps_cfg = pps_cfg
    model_cfg = cfg.model_cfg
    model = create_model(model_cfg, meta, model_meta)
    log.info('Model is prepared successfully')
    return model, model_meta


