import tempfile
import os
import mlflow
import json
import torch
import copy

import model_src.train
from model_src.data.dataset_builders.builder import build_data, update_train_data, update_dl_cfg
from model_src.models.model_builder import prepare_fine_tune_model, update_model_cfg, build_predictor
from model_src.models.post_processor import build_post_processor

from model_src.eval import evaluate
from model_src.prepare_train.prepare_train import prepare_train_params, update_train_cfg

from app_src.schemas.data import DatasetJobRequest
from app_src.schemas.models import ModelJobRequest
from app_src.schemas.train import TrainJobRequest, PostProcessingJobRequest

from common.logger import get_logger

log = get_logger(__name__)

mlflow_public_uri = os.getenv("MLFLOW_PUBLIC_URI")

# TODO: should enable user to load the cfg based on the history run
def train_model(predictor, train, val, meta, dl_cfg, model_cfg, train_cfg, cfg):
    log.info('Initializing training model process')
    if predictor is None or train is None:
        log.error('Model training can not be initalized because predictor or dataset is not loaded')
        raise AssertionError(f'Model and Dataset need to be prepared first')
    
    exp_name = cfg.exp_name
    log.info(f'Setting experiment name: {exp_name}')
    mlflow.set_experiment(exp_name)

    run_name = cfg.run_name
    log.info(f'Starting run "{run_name}" for experiment "{exp_name}"')
    with mlflow.start_run(run_name=run_name):
        predictor = model_src.train.train_model(predictor, train, val, meta, cfg)
        store_artifacts(predictor.get_model(), meta, dl_cfg, model_cfg, cfg)
    mlflow.end_run()
    log.info('Training model process is successfully finished')

def inspect_run(client, data):
    log.info('Initializing inspect process')
    run_id = data.run_id

    log.info(f'Retrieving logged artifacts for run_id: {run_id}')
    data = retrieve_logged_artficats(client, run_id)

    log.info('Rebuilding Dataloaders')
    train_dl, val_dl, test_dl, meta = build_data(DatasetJobRequest(**data['dataset_cfg']))

    log.info('Rebuilding predictor')
    model_cfg = ModelJobRequest(**data['model_cfg'])
    pp_cfg = PostProcessingJobRequest(**data['pp_cfg']) if data['pp_cfg'] is not None else None
    predictor = build_predictor(model_cfg, pp_cfg)
    predictor.get_model().load_state_dict(data['model_state_data'])

    log.info('Rebuilding train params')
    train_params = prepare_train_params(predictor.get_model_parameters(), meta, TrainJobRequest(**data['train_cfg']).train_cfg)

    log.info('Evaluating run')
    test_metrics, dict_error_analysis = evaluate(predictor, val_dl, train_params, collect_error_analysis=True)
    
    result = {
        'val_metrics': dict(test_metrics),
        'val_error_table': dict_error_analysis['df'],
    }
    if 'confusion_matrix' in dict_error_analysis:
        log.info('Adding confusion matrix in the result')
        result['val_confusion_matrix'] = dict_error_analysis['confusion_matrix']
    result['layer_cfg'] = model_cfg.model_dump(exclude={'model_type'})
                                               
    ctx_dict = {
        'predictor': predictor,
        'train': train_dl,
        'val': val_dl,
        'test': test_dl,
        'meta': meta,
        'cached_dl_cfg': DatasetJobRequest(**data['dataset_cfg']),
        'cached_model_cfg': model_cfg,
        'cached_train_cfg': TrainJobRequest(**data['train_cfg']),
        'cached_pp_cfg': pp_cfg,
        'cached_run_id': run_id,
    }
    log.info('Inspect process is successfully finished')
    return result, ctx_dict

def post_process(predictor, val_dl, meta, dl_cfg, model_cfg, train_cfg, pp_cfg, run_id):
    log.info('Initiazling post processsor process')
    if predictor is None or val_dl is None:
        log.error('Post Processing can not be initalized because predictor or dataset is not loaded')
        raise AssertionError(f'Model and Dataset need to be prepared first')
    
    log.info('Rebuilding train params')
    train_params = prepare_train_params(predictor.get_model_parameters(), meta, train_cfg.train_cfg)
    device = train_params.device

    log.info('Initializing post processor')
    pp = build_post_processor(pp_cfg)

    log.info('Initializing training of post processor parameters')
    pp.train(copy.deepcopy(predictor.get_model()), val_dl, device)
    
    log.info('Overriding current post processor')
    predictor.set_pp(pp)

    updated_pp_cfg = predictor.get_pp().get_cfg()
    
    exp_name = train_cfg.exp_name
    mlflow.set_experiment(exp_name)
    log.info(f'Starting run "{pp_cfg.new_run_name}" for experiment "{exp_name}"')
    with mlflow.start_run(run_name=pp_cfg.new_run_name):
        val_metrics, dict_error_analysis = evaluate(predictor, val_dl, train_params, collect_error_analysis=True)

        mlflow.log_metrics({f'val_{name.lower()}': metric_val for name, metric_val in val_metrics})
        store_artifacts(predictor.get_model(), meta, dl_cfg, model_cfg, train_cfg, updated_pp_cfg)
        
        exp_id = mlflow.active_run().info.experiment_id
        updated_run_id = mlflow.active_run().info.run_id
        parent_url = f'{mlflow_public_uri}/#/experiments/{exp_id}/runs/{run_id}'
        mlflow.set_tag('Parent_run_id', parent_url)
        mlflow.set_tag('run_type', 'post_processing')
    mlflow.end_run()

    result = {
        'val_metrics': dict(val_metrics),
        'val_error_table': dict_error_analysis['df'],
    }
    if 'confusion_matrix' in dict_error_analysis:
        log.info('Adding confusion matrix in the result')
        result['val_confusion_matrix'] = dict_error_analysis['confusion_matrix']

    result['layer_cfg'] = model_cfg.model_dump(exclude={'model_type'})

    ctx_dict = {
        'cached_pp_cfg': updated_pp_cfg,
        'cached_run_id': updated_run_id
    }
    log.info('Post processor is successfully finished')
    return result, ctx_dict

def fine_tune(predictor, train_dl, val_dl, meta, dl_cfg, model_cfg, train_cfg, pp_cfg, ft_cfg, run_id):
    log.info('Initiazling fine tune process')
    if predictor is None or train_dl is None:
        log.error('Post Processing can not be initalized because predictor or dataset is not loaded')
        raise AssertionError(f'Model and Dataset need to be prepared first')
    
    # TODO: need to be careful that input size is not changed
    log.info('Updating train Dataloader')
    new_ds_cfg =ft_cfg.new_ds_cfg
    train_dl = update_train_data(train_dl, meta, new_ds_cfg)
    updated_dl_cfg = update_dl_cfg(dl_cfg, new_ds_cfg)

    log.info('Updating model layers')
    new_layers_cfg = ft_cfg.new_layers_cfg
    predictor = prepare_fine_tune_model(predictor, new_layers_cfg)
    updated_model_cfg = update_model_cfg(model_cfg, new_layers_cfg)

    log.info('Updating train params')
    new_train_cfg = ft_cfg.new_train_cfg
    updated_train_cfg = update_train_cfg(new_train_cfg, train_cfg)

    log.info('Disabling the post processor')
    predictor.set_pp(None)

    exp_name = train_cfg.exp_name
    mlflow.set_experiment(exp_name)
    log.info(f'Starting run "{pp_cfg.new_run_name}" for experiment "{exp_name}"')
    with mlflow.start_run(run_name=ft_cfg.new_run_name):
        log.info('Starting the fine-tune training')
        predictor = model_src.train.train_model(predictor, train_dl, val_dl, meta, updated_train_cfg)
        
        run_info = mlflow.active_run().info
        exp_id = run_info.experiment_id
        parent_url = f'{mlflow_public_uri}/#/experiments/{exp_id}/runs/{run_id}'
        mlflow.set_tag('run_type', 'Fine tuning')
        mlflow.set_tag('Parent_run_id', parent_url)
        if pp_cfg is not None:
            mlflow.set_tag('is_post_processed_parent', 'true')
        else:
            mlflow.set_tag('is_post_processed_parent', 'false')
        store_artifacts(predictor.get_model(), meta, updated_dl_cfg, updated_model_cfg, train_cfg, pp_cfg)
    mlflow.end_run()
    log.info('Fine tune is successfully finished')

def store_artifacts(model, meta, dl_cfg, model_cfg, train_cfg, pp_cfg=None):
    log.info('Initiazling artifacts storing')
    with tempfile.TemporaryDirectory() as td:
        log_dataset(dl_cfg, meta, td)
        log_model(model, model_cfg, td)
        log_post_processor(pp_cfg, td)
        log_train_cfg(train_cfg, td)
    log.info('Artifacts stored successfully')

def log_dataset(dl_cfg, meta, td):
    ds_path = os.path.join(td, 'data_cfg.json')
    with open(ds_path, 'w') as f:
        data_dict = {
        'data_cfg': dl_cfg.model_dump(),
        'meta': meta.to_dict()
        }   
        json.dump(data_dict, f, indent=2)
    log.info(f'storing dataset artifact on the path: {ds_path}')
    mlflow.log_artifact(ds_path, artifact_path='data')

def log_model(model, model_cfg, td):
    model_cfg_path = os.path.join(td, 'model_cfg.json')
    with open(model_cfg_path, 'w') as f:
        json.dump(model_cfg.model_dump(), f, indent=2)
    log.info(f'storing model cfg artifact on the path: {model_cfg_path}')
    mlflow.log_artifact(model_cfg_path, artifact_path='model')

    model_path = os.path.join(td, 'model.pt')
    torch.save(model.state_dict(), model_path)
    log.info(f'storing model state artifact on the path: {model_path}')
    mlflow.log_artifact(model_path, artifact_path='model')

def log_post_processor(pp_cfg, td):
    pp_path = os.path.join(td, 'pp_cfg.json')
    with open(pp_path, 'w') as f:
        if pp_cfg is None:
            json.dump(None, f)
        else:
            json.dump(pp_cfg.model_dump(), f, indent=2)
    log.info(f'Storing post processor artifact on the path: {pp_path}')
    mlflow.log_artifact(pp_path, artifact_path='model')

def log_train_cfg(train_cfg, td):
    train_path = os.path.join(td, 'train_cfg.json')
    with open(train_path, 'w') as f:
        json.dump(train_cfg.model_dump(), f, indent=2)
    log.info(f'storing train params artifact on the path: {train_path}')
    mlflow.log_artifact(train_path, artifact_path='train')

def retrieve_logged_artficats(client, run_id):
    model_path = client.download_artifacts(
        run_id,
        'model/model.pt'
    )
    model_cfg_path = client.download_artifacts(
        run_id,
        'model/model_cfg.json'
    )
    pp_cfg_path = client.download_artifacts(
        run_id,
        'model/pp_cfg.json'
    )
    dataset_cfg_path = client.download_artifacts(
        run_id,
        'data/data_cfg.json'
    )
    train_cfg_path = client.download_artifacts(
        run_id,
        'train/train_cfg.json'
    )
    model_state_data = torch.load(model_path, map_location="cpu")
    with open(model_cfg_path) as f:
        model_cfg = json.load(f)
    with open(pp_cfg_path) as f:
        pp_cfg = json.load(f)
    with open(dataset_cfg_path) as f:
        data = json.load(f)
        data_cfg = data['data_cfg']
        meta = data['meta']
    with open(train_cfg_path) as f:
        train_cfg = json.load(f)
    run = client.get_run(run_id)
    params = run.data.params
    # metrics = run.data.metrics
    return {
        'model_state_data': model_state_data,
        'model_cfg': model_cfg,
        'dataset_cfg': data_cfg,
        'meta': meta,
        'train_cfg': train_cfg,
        'params': params,
        # 'metrics': metrics,
        'pp_cfg': pp_cfg
    }