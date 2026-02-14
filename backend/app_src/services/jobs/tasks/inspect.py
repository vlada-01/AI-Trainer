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

from app_src.services.jobs.reader_writer import ArtifactReader

from common.logger import get_logger

log = get_logger(__name__)

mlflow_public_uri = os.getenv("MLFLOW_PUBLIC_URI")

def atomic_inspect_run(data):
    log.info('Initializing inspect process')
    run_id = data.run_id

    log.info(f'Retrieving logged artifacts for run_id: {run_id}')
    with ArtifactReader(client, run_id) as r:
        data = r.get_data()
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