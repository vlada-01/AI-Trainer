from model_src.models.model_builder import build_predictor

from common.logger import get_logger

log = get_logger(__name__)


def atomic_prepare_predictor(cfg):
    log.info('Initializing prepare_predictor process')
    predictor = build_predictor(cfg)
    log.info('Prepare Predictor is successfully finished')
    ctx_dict = {
        'predictor': predictor,
        'cached_model_cfg': cfg
    }
    result = ''
    return result, ctx_dict