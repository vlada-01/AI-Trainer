from packages.train_lib.prepare_data.data_builder import build_data
from packages.train_lib.prepare_model.model_builder import build_model
from packages.train_lib.prepare_train.train_builder import prepare_engine
from packages.train_lib.prepare_model.models.model.model import update_model_pps

import train_server.schemas.job_request as requests

from packages.mlflow_logger.reader import MlflowReader

from packages.logger.logger import get_logger

log = get_logger(__name__)

# TODO: Need to test All transformation types
# FIXME:need to update transformations so that each head can be independent
def atomic_prepare_dataset(cfg):
    log.info('Initializing prepare dataset process')
    train, val, test, data_meta =  build_data(cfg)
    ctx_dict = {}
    ctx_dict = update_ctx_dict(ctx_dict, 'cfgs', dl_cfg=cfg)
    ctx_dict = update_ctx_dict(ctx_dict, 'runtime', train=train, val=val, test=test)
    ctx_dict = update_ctx_dict(ctx_dict, 'meta', data_meta=data_meta)
    results = {
        'sample_size': data_meta.get_necessary_sizes(),
        'input_keys': data_meta.get_input_keys()
    }
    log.info('Prepare dataset is successfullty finished')
    return results, ctx_dict

# TODO: need to add check for input/output size connections
# TODO: need to check if component cfg is working
def atomic_prepare_model(meta, cfg):
    log.info('Initializing prepare model process')
    model, model_meta = build_model(meta, cfg)
    ctx_dict = {}
    ctx_dict = update_ctx_dict(ctx_dict, 'cfgs', model_cfg=cfg)
    ctx_dict = update_ctx_dict(ctx_dict, 'runtime', model=model)
    ctx_dict = update_ctx_dict(ctx_dict, 'meta', model_meta=model_meta)
    result = 'Model is successfully prepared'
    log.info('Prepare Model is successfully finished')
    return result, ctx_dict

def atomic_prepare_engine(model, train, val, test, meta, train_cfg):
    log.info('Initializing prepare engine manager process')
    engine, train_meta = prepare_engine(train_cfg, model, train, val, test, meta)
    ctx_dict = {}
    ctx_dict = update_ctx_dict(ctx_dict, 'cfgs', train_cfg=train_cfg)
    ctx_dict = update_ctx_dict(ctx_dict, 'runtime', engine=engine)
    ctx_dict = update_ctx_dict(ctx_dict, 'meta', train_meta=train_meta)
    result = 'Train Params are successfully prepared'
    log.info('Prepare engine manager process is successfully finished')    
    return result, ctx_dict

def atomic_prepare_complete_train(meta, cfgs):
    log.info('Initiazling prepare train with all configurations')
    _, ctx_dict_1 = atomic_prepare_dataset(cfgs.dataset_cfg)
    
    data_meta = ctx_dict_1['meta']['data_meta']
    meta.set('data_meta', data_meta)
    train = ctx_dict_1['runtime']['train']
    val = ctx_dict_1['runtime']['val']

    _, ctx_dict_2 = atomic_prepare_model(meta, cfgs.model_cfg)
    model = ctx_dict_2['runtime']['model']
    model_meta = ctx_dict_2['meta']['model_meta']
    meta.set('model_meta', model_meta)
    
    _, ctx_dict_3 = atomic_prepare_engine(model, train, val, meta, cfgs.train_cfg)
    train_meta = ctx_dict_3['meta']['train_meta']
    meta.set('train_meta', train_meta)
    
    ctx_dict = merge(ctx_dict_1, ctx_dict_2)
    ctx_dict = merge(ctx_dict, ctx_dict_3)
    result = 'Configurations are successfully prepared'
    log.info('Prepare train process is successfully finished')
    return result, ctx_dict

# TODO: add later support to load all metas directly
# FIXME: right now, meta needs to be retrieved to get specs
def atomic_prepare_default_from_run(meta, cfg, job_id):
    log.info('Initiazling prepare train with configurations from run')
    run_id = cfg.run_id
    reader = MlflowReader(job_id, run_id)

    with reader.open_artifact_reader() as r:
        ds_cfg = r.log_cfg(rel_path='cfgs/dataset_cfg.json')
        model_cfg = r.log_cfg(rel_path='cfgs/model_cfg.json')
        train_cfg = r.log_cfg(rel_path='cfgs/train_cfg.json')
        model_state_dict = r.load_model_state()

        # FIXME: needs to be udpated
        # meta = r.load_meta()
    
    cfgs = requests.PrepareCompleteTrainJobRequest(
        dataset_cfg=requests.PrepareDatasetJobRequest(**ds_cfg),
        model_cfg=requests.PrepareModelJobRequest(**model_cfg),
        train_cfg=requests.PrepareTrainJobRequest(**train_cfg)
    )
    _, ctx_dict = atomic_prepare_complete_train(meta, cfgs)
    ctx_dict['runtime']['model'].load_state_dict(model_state_dict)
    
    ctx_dict = update_ctx_dict(ctx_dict, 'runtime', {'mlflow_run_id': run_id})
    result = 'Configurations from run are successfully prepared'
    log.info('Prepare train with configurations from run is successfully finished')
    return result, ctx_dict

def atomic_prepare_post_process(engine, meta, cached_model_cfg, pps_cfg):
    log.info('Initiazling post processsor process')
    
    log.info('Initializing post processor')
    model = engine.get_model()
    update_model_pps(model, meta, pps_cfg)
    cached_model_cfg.pps_cfg = pps_cfg

    log.info('Initializing training of post processor parameters')
    engine.train_pp()

    result = 'Post Processor is successfully prepared'
    ctx_dict = {}
    ctx_dict = update_ctx_dict(ctx_dict, 'cfgs', {'model_cfg': cached_model_cfg})

    log.info('Post processor process is successfully finished')
    return result, ctx_dict

def update_ctx_dict(ctx_dict, key, **kwargs):
    ctx_dict.setdefault(key, {}).update(kwargs)
    return ctx_dict

def merge(ctx_dict1, ctx_dict2):
    for k in ctx_dict1.keys():
        ctx_dict1[k] = {**ctx_dict1[k], **ctx_dict2}
    return ctx_dict1