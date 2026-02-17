import copy
from model_src.data.dataset_builders.builder import build_data
from model_src.models.model_builder import build_predictor
from model_src.models.post_processor import build_post_processor
from model_src.prepare_train.prepare_train import prepare_train_params

import app_src.schemas.job_request as requests

from app_src.services.reader_writer import ArtifactReader

from common.logger import get_logger

log = get_logger(__name__)


# TODO: Need to test All transformation types
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
        'sample_size': meta.get_necessary_sizes(),
        'input_keys': meta.get_input_keys()
    }
    log.info('Prepare dataset is successfullty finished')
    return result, ctx_dict

# TODO: need to add check for input/output size connections
# TODO: need to check if component cfg is working
def atomic_prepare_predictor(cfg):
    log.info('Initializing prepare predictor process')
    predictor = build_predictor(cfg)
    ctx_dict = {
        'predictor': predictor,
        'cached_model_cfg': cfg
    }
    result = 'Predictor is successfully prepared'
    log.info('Prepare Predictor is successfully finished')
    return result, ctx_dict

def atomic_prepare_train_params(predictor, meta, train_cfg):
    log.info('Initializing prepare training parameters process')
    model = predictor.get_model()
    train_params = prepare_train_params(model.parameters(), meta, train_cfg)

    ctx_dict = {
        'train_params': train_params,
        'cached_train_cfg': train_cfg,
    }
    result = 'Train Params are successfully prepared'
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
    
    result = 'Configurations are successfully prepared'
    ctx_dict = {
        **ctx_dict_1,
        **ctx_dict_2,
        **ctx_dict_3
    }
    log.info('Prepare train process is successfully finished')
    return result, ctx_dict

def atomic_prepare_default_from_run(cfg, job_id):
    log.info('Initiazling prepare train with configurations from run')
    run_id = cfg.run_id

    with ArtifactReader(job_id, run_id) as r:
        ds_cfg = r.load_data_cfg()
        # TODO: add later meta, when atomic_prepare_dataset is updated
        model_cfg = r.load_model_cfg()
        model_state_dict = r.load_model_state()
        train_cfg = r.load_train_cfg()
    
    cfgs = requests.PrepareCompleteTrainJobRequest(
        dataset_cfg=requests.PrepareDatasetJobRequest(**ds_cfg),
        model_cfg=requests.PrepareModelJobRequest(**model_cfg),
        train_cfg=requests.PrepareTrainJobRequest(**train_cfg)
    )
    _, ctx_dict = atomic_prepare_complete_train(cfgs)
    ctx_dict['predictor'] = ctx_dict['predictor'].get_model().load_state_dict(model_state_dict)
    
    ctx_dict = {
        **ctx_dict,
        'cached_mlflow_run_id': run_id
    }
    result = 'Configurations from run are successfully prepared'
    log.info('Prepare train with configurations from run is successfully finished')
    return result, ctx_dict

def atomic_prepare_post_process(predictor, val_dl, train_params, pp_cfg):
    log.info('Initiazling post processsor process')
    
    log.info('Initializing post processor')
    pp = build_post_processor(pp_cfg)

    log.info('Initializing training of post processor parameters')
    device = train_params.device
    pp.train(copy.deepcopy(predictor.get_model()), val_dl, device)
    
    log.info('Setting post processor in the predictor')
    predictor.set_pp(pp)

    updated_pp_cfg = predictor.get_pp().get_cfg()

    result = 'Post Processor is successfully prepared'
    ctx_dict = {
        'predictor': predictor,
        'cached_pp_cfg': updated_pp_cfg
    }

    log.info('Post processor is successfully finished')
    return result, ctx_dict