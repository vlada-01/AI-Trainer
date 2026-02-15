from model_src.eval import predict

from common.logger import get_logger

log = get_logger(__name__)

def atomic_final_eval(predictor, test, train_params, parent_id):
    log.info('Initializing predict process')
    if parent_id is None:
        raise RuntimeError('Cannot initialize post processing wihtout parent mlflow run id')
    device = train_params.device
    metrics = train_params.metrics
    error_analysis = train_params.error_analysis
    test_metrics, dict_error_analysis = predict(predictor, test, device, metrics, error_analysis)
    
    # TODO: update result, ctx_dict
    result = {
        'test_metrics': dict(test_metrics),
        'test_error_table': dict_error_analysis['df'],
    }
    if 'confusion_matrix' in dict_error_analysis:
        log.info('Adding confusion matrix in the result')
        result['test_confusion_matrix'] = dict_error_analysis['confusion_matrix']
    ctx_dict = {}
    log.info('Predict proccess is succesfully finished')
    return result, ctx_dict