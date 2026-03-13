from packages.core_lib.meta import create_meta
from packages.core_lib.models.model.model import create_model

from packages.logger.logger import get_logger

log = get_logger(__name__)

def build_model(meta, cfg):
    log.info('Initializing model builder')
    
    log.info('Initializing ModelMeta')
    model_meta_cfg = cfg.model_meta_cfg
    model_meta = create_meta(model_meta_cfg)
    
    model_cfg = cfg.model_config
    model = create_model(model_cfg, meta, model_meta)
    log.info('Model is prepared successfully')
    return model, model_meta


