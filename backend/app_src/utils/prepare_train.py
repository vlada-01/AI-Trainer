from model_src.prepare_train.prepare_train import prepare_train_params

from common.logger import get_logger

log = get_logger(__name__)

def atomic_prepare_train(model, meta, train_cfg):
    log.info('Initializing prepare training parameters process')
    train_params = prepare_train_params(model.parameters(), meta, train_cfg)
    log.info('Prepare training parameters process is successfully finished')
    ctx_dict = {
        'train_params': train_params,
        'cached_train_cfg': train_cfg
    }
    result = ''
    
    return result, ctx_dict