from model_src.data.dataset_builders.builder import build_data
from model_src.models.model_builder import build_predictor
from model_src.prepare_train.prepare_train import prepare_train_params

import app_src.schemas.job_request as requests

from app_src.services.reader_writer import ArtifactReader

from common.logger import get_logger

log = get_logger(__name__)


# TODO: Need to test All transformation types
# TODO: If dataset does not provide validation set, need to manually split train dataset
# TODO: need to add support to load meta by cfg
def atomic_prepare_dataset(cfg):
    log.info('Initializing prepare dataset process')
    train, val, _, meta =  build_data(cfg)
    ctx_dict = {
        'train': train,
        'val': val,
        'meta': meta,
        'cached_dl_cfg': cfg
    }
    result = {
        'sample_size': meta.get_necessary_sizes()
    }
    log.info('Prepare dataset is successfullty finished')
    return result, ctx_dict

# TODO: need to add check for input/output size connections
# TODO: need to check for other layer types
# TODO: needs to return back to add some more input types...
def atomic_prepare_predictor(cfg):
    log.info('Initializing prepare predictor process')
    predictor = build_predictor(cfg)
    ctx_dict = {
        'predictor': predictor,
        'cached_model_cfg': cfg
    }
    result = ''
    log.info('Prepare Predictor is successfully finished')
    return result, ctx_dict

def atomic_prepare_train_params(predictor, meta, train_cfg):
    # TODO: prevent if data and predictor are not iniialized
    log.info('Initializing prepare training parameters process')
    model = predictor.get_model()
    train_params = prepare_train_params(model.parameters(), meta, train_cfg)

    ctx_dict = {
        'train_params': train_params,
        'cached_train_cfg': train_cfg,
    }
    result = ''
    log.info('Prepare training parameters process is successfully finished')    
    return result, ctx_dict

# TODO: maybe add pp as well
def atomic_prepare_complete_train(cfgs):
    log.info('Initiazling prepare train with all configurations')
    ds_cfg, model_cfg, train_cfg = cfgs.dataset_cfg, cfgs.model_cfg, cfgs.train_cfg
    _, ctx_dict_1 = atomic_prepare_dataset(ds_cfg)
    meta = ctx_dict_1['meta']
    
    _, ctx_dict_2 = atomic_prepare_predictor(model_cfg)
    predictor = ctx_dict_2['predictor']
    
    _, ctx_dict_3 = atomic_prepare_train_params(predictor, meta, train_cfg)
    
    result = ''
    ctx_dict = {
        **ctx_dict_1,
        **ctx_dict_2,
        **ctx_dict_3
    }
    log.info('Prepare train process is successfully finished')
    return result, ctx_dict

def atomic_prepare_cfg_from_run(cfg, job_id):
    log.info('Initiazling prepare train with configurations from run')
    run_id = cfg.run_id

    with ArtifactReader(job_id, run_id) as r:
        ds_cfg = r.load_data_cfg()
        model_cfg = r.load_data_cfg()
        train_cfg = r.load_data_cfg()
    
    cfgs = requests.PrepareCompleteTrainJobRequest(
        dataset_cfg=requests.PrepareDatasetJobRequest(**ds_cfg),
        model_cfg=requests.PrepareModelJobRequest(**model_cfg),
        train_cfg=requests.PrepareTrainJobRequest(**train_cfg)
    )
    result, ctx_dict = atomic_prepare_complete_train(cfgs)
    
    ctx_dict = {
        **ctx_dict,
        'cached_mlflow_run_id': run_id
    }
    log.info('Prepare train with configurations from run is successfully finished')
    return result, ctx_dict
