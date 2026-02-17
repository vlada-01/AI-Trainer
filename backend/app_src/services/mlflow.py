import os
from uuid import uuid4
from app_src.schemas.mlflow import Experiment

from app_src.services.reader_writer import ArtifactReader

mlflow_public_uri = os.getenv("MLFLOW_PUBLIC_URI", "http://localhost:5000")

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

def get_run_results(client, run_id):
    job_id = uuid4().hex
    log.info('Initializing inspect process')

    log.info(f'Retrieving logged artifacts for run_id: {run_id}')
    with ArtifactReader(job_id, run_id) as r:
        error_analysis_dict = r.load_error_analysis()

    run = client.get_run(run_id)
    metrics = run.data.metrics
    # TODO: watch out of types
    results_dict = {
        'val_metrics': metrics,
        'val_error_table': error_analysis_dict['df']
    }
    if 'confusion_matrix' in error_analysis_dict:
        log.info('Adding confusion matrix in the result')
        results_dict['val_confusion_matrix'] = error_analysis_dict['confusion_matrix']
    log.info('Inspect process is successfully finished')
    return results_dict


