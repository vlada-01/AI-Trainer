from model_src.models.model_builder import build_predictor

from common.logger import get_logger

log = get_logger(__name__)

# TODO: needs to return back to add some more input types...
def prepare_predictor(cfg):
    log.info('Initializing prepare_predictor process')
    predictor = build_predictor(cfg)
    log.info('Prepare Predictor is successfully finished')
    return {
        'predictor': predictor,
        'cfg': cfg
    }

def predict(data):
    pass