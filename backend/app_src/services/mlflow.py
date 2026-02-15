from uuid import uuid4
from app_src.schemas.mlflow import Experiment

from app_src.services.reader_writer import ArtifactReader

from app_src.app import mlflow_public_uri

from common.logger import get_logger

log = get_logger(__name__)


def get_experiments(client):
    exps = client.search_experiments()
    active_exps = [e for e in exps if e.lifecycle_stage == 'active']
    list_of_exps = [
        Experiment(
            name=e.name,
            url=f'{mlflow_public_uri}/#/experiments/{e.experiment_id}'
        )
        for e in active_exps
    ]
    return list_of_exps

def get_run_results(run_id):
    job_id = uuid4().hex
    log.info('Initializing inspect process')

    log.info(f'Retrieving logged artifacts for run_id: {run_id}')
    with ArtifactReader(job_id, run_id) as r:
        val_metrics = r.load_metrics()
        val_error_analysis = r.load_val_error_analysis()
    
    # TODO: watch out of types
    result = {
        'val_metrics': val_metrics,
        'val_error_table': val_error_analysis['df']
    }
    if 'confusion_matrix' in val_error_analysis:
        log.info('Adding confusion matrix in the result')
        result['val_confusion_matrix'] = val_error_analysis['confusion_matrix']
    ctx_dict = {}
    log.info('Inspect process is successfully finished')
    return result, ctx_dict

# TODO: should enable user to load the cfg based on the history run
# TODO: add atomic function for getting the configurations


