from model_src.models.model_builder import build_predictor

from common.logger import get_logger

log = get_logger(__name__)

# TODO: needs to return back to add some more configuration types...
def prepare_predictor(cfg):
    log.info(f'Initialzing predictor preparation with: {cfg.model_dump()}')
    predictor = build_predictor(cfg)
    return {
        'predictor': predictor,
        'cfg': cfg
    }

def predict(data):
    pass