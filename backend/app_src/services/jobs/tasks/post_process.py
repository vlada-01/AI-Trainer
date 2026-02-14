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

from backend.app_src.schemas.data import DatasetJobRequest
from app_src.schemas.models import ModelJobRequest
from app_src.schemas.train import TrainJobRequest, PostProcessingJobRequest

from common.logger import get_logger

log = get_logger(__name__)

def atomic_post_process(predictor, val_dl, meta, dl_cfg, model_cfg, train_cfg, pp_cfg, run_id):
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