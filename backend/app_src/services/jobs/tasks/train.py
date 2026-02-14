import os
import mlflow

from  model_src.train import train_model

from app_src.services.jobs.reader_writer import ArtifactWriter

from common.logger import get_logger

log = get_logger(__name__)

# TODO: should enable user to load the cfg based on the history run
# TODO: update train_model function to get already prepared train_params
def atomic_train_model(predictor, train, val, meta, dl_cfg, model_cfg, train_cfg, job_id):
    log.info('Initializing training model process')
    if predictor is None or train is None:
        log.error('Model training can not be initalized because predictor or dataset is not loaded')
        raise AssertionError(f'Model and Dataset need to be prepared first')
    
    exp_name = train_cfg.exp_name
    log.info(f'Setting experiment name: {exp_name}')
    mlflow.set_experiment(exp_name)

    run_name = train_cfg.run_name
    log.info(f'Starting run "{run_name}" for experiment "{exp_name}"')
    with mlflow.start_run(run_name=run_name):
        predictor = train_model(predictor, train, val, meta, train_cfg)
        run_id = mlflow.active_run().info.run_id
        with ArtifactWriter(job_id, run_id) as w:
            w.save_data_cfg(dl_cfg.model_dump())
            w.save_meta(meta.to_dict())
            w.save_model_cfg(model_cfg.model_dump())
            w.save_model_state(predictor.get_model().state_dict())
            # TODO: fix me, not pp_cfg, but pp is passed
            w.save_post_processor_cfg(predictor.get_pp())
            w.save_train_cfg(train_cfg.model_dump())
            w.log_artifacts()
    mlflow.end_run()
    log.info('Training model process is successfully finished')