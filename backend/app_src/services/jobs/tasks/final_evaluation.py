from model_src.data.dataset_builders.builder import build_data
from model_src.models.model_builder import build_predictor
from model_src.prepare_train.prepare_train import prepare_train_params
import model_src.eval

import app_src.schemas.job_request as requests

from common.logger import get_logger

log = get_logger(__name__)

def atomic_final_eval(client, req):
    log.info('Initializing predict process')
    run_id = req.run_id

    log.info(f'Retrieving logged artifacts for run_id: {run_id}')
    data = retrieve_logged_artficats(client, run_id)

    log.info('Rebuilding test Dataloader')
    _, _, test_dl, meta = build_data(requests.PrepareDatasetJobRequest(**data['dataset_cfg']))

    log.info('Rebuilding predictor')
    model_cfg = requests.PrepareModelJobRequest(**data['model_cfg'])
    pp_cfg = requests.PreparePostProcessingJobRequest(**data['pp_cfg']) if data['pp_cfg'] is not None else None
    predictor = build_predictor(model_cfg, pp_cfg)
    predictor.get_model().load_state_dict(data['model_state_data'])

    log.info('Rebuilding train params')
    train_params = prepare_train_params(predictor.get_model_parameters(), meta, requests.PrepareTrainJobRequest(**data['train_cfg']).train_cfg)
    device = train_params.device
    metrics = train_params.metrics
    error_analysis = train_params.error_analysis

    log.info('Last moment to pray!')
    test_metrics, dict_error_analysis = model_src.eval.predict(predictor, test_dl, device, metrics, error_analysis)
    
    # TODO: update result, ctx_dict
    result = {
        'test_metrics': dict(test_metrics),
        'test_error_table': dict_error_analysis['df'],
    }
    if 'confusion_matrix' in dict_error_analysis:
        log.info('Adding confusion matrix in the result')
        result['test_confusion_matrix'] = dict_error_analysis['confusion_matrix']
    log.info('Predict proccess is succesfully finished')
    return result, ctx_dict