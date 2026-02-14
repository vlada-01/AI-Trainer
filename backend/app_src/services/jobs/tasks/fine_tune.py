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

mlflow_public_uri = os.getenv("MLFLOW_PUBLIC_URI")


def atomic_fine_tune(predictor, train_dl, val_dl, meta, dl_cfg, model_cfg, train_cfg, pp_cfg, ft_cfg, run_id):
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