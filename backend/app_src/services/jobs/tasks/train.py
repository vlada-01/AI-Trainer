import mlflow

from  model_src.train import train_model
from model_src.eval import evaluate

from app_src.services.reader_writer import ArtifactWriter

from common.logger import get_logger

log = get_logger(__name__)

def atomic_train_model(predictor, train, val, meta, train_params, dl_cfg, model_cfg, train_cfg, data, job_id):
    log.info('Initializing training model process')
    exp_name = data.exp_name
    log.info(f'Setting experiment name: {exp_name}')
    mlflow.set_experiment(exp_name)

    #TODO: data.model_name is not used

    run_name = data.run_name
    log.info(f'Starting run "{run_name}" for experiment "{exp_name}"')
    with mlflow.start_run(run_name=run_name):
        predictor = train_model(predictor, train, train_params)
        _, val_error_analysis_dict = evaluate(predictor, val, train_params, collect_error_analysis=True)
        run_id = mlflow.active_run().info.run_id
        with ArtifactWriter(job_id, run_id) as w:
            w.save_data_cfg(dl_cfg.model_dump())
            w.save_meta(meta.to_dict())
            w.save_model_cfg(model_cfg.model_dump())
            w.save_model_state(predictor.get_model().state_dict())
            w.save_post_processor_cfg(None)
            w.save_train_cfg(train_cfg.model_dump())
            w.save_error_analysis(val_error_analysis_dict)
            w.log_artifacts()
    mlflow.end_run()
    result = ''
    ctx_dict = {}
    log.info('Training model process is successfully finished')
    return result, ctx_dict