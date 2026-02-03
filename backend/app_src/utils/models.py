from model_src.models.model_builder import build_model

from common.logger import get_logger

log = get_logger(__name__)

# TODO: needs to return back to add some more configuration types...
def prepare_model(cfg):
    log.info(f'Initialzing model preparation with: {cfg.model_dump()}')
    model = build_model(cfg)
    return {
        'model': model,
        'cfg': cfg
    }

def predict(data):
    pass