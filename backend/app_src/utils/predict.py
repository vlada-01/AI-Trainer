
from model_src.data.dataset_builders.builder import build_data
from model_src.models.model_builder import build_model
from model_src.models.post_processor import build_post_processor
from model_src.prepare_train.prepare_train import prepare_train_params
import model_src.eval

from app_src.utils.train import retrieve_logged_artficats

from app_src.schemas.data import DatasetJobRequest
from app_src.schemas.models import ModelJobRequest
from app_src.schemas.train import TrainJobRequest, PostProcessingJobRequest

from common.logger import get_logger

log = get_logger(__name__)

def predict(client, req):
    log.info('Initializing test process')
    run_id = req.run_id

    data = retrieve_logged_artficats(client, run_id)

    # TODO: make meta be used later on
    log.info('Rebuilding test Dataloader')
    _, _, test_dl, meta = build_data(DatasetJobRequest(**data['dataset_cfg']))

    log.info('Rebuilding model')
    model = build_model(ModelJobRequest(**data['model_cfg']))
    model.load_state_dict(data['model_state_data'])

    log.info('Rebuilding test params')
    train_params = prepare_train_params(model.parameters(), meta, TrainJobRequest(**data['train_cfg']).train_cfg)
    device = train_params.device
    metrics = train_params.metrics
    error_analysis = train_params.error_analysis

    log.info('Rebuilding the PostProcessor for test')
    if data['pp_cfg'] is not None:
        post_processor = build_post_processor(PostProcessingJobRequest(**data['pp_cfg']))
    else:
        log.warning('No post processor')
        post_processor = None

    log.info('Last moment to pray!')
    test_metrics, dict_error_analysis = model_src.eval.predict(model, test_dl, device, metrics, error_analysis, post_processor=post_processor)
    
    result = {
        'test_metrics': dict(test_metrics),
        'test_error_table': dict_error_analysis['df'],
    }
    if 'confusion_matrix' in dict_error_analysis:
        log.info('Adding confusion matrix in the result')
        result['test_confusion_matrix'] = dict_error_analysis['confusion_matrix']

    return result

    