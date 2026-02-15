import os
import mlflow
from model_src.eval import predict

from app_src.services.reader_writer import ArtifactWriter

from common.logger import get_logger

log = get_logger(__name__)

mlflow_public_uri = os.getenv("MLFLOW_PUBLIC_URI", "http://localhost:5000")

def atomic_final_eval(predictor, test, meta, train_params, dl_cfg, model_cfg, train_cfg, pp_cfg, parent_id, data, job_id):
    log.info('Initializing final evaluation process')
    exp_name = train_cfg.exp_name
    log.info(f'Setting experiment name: {exp_name}')
    mlflow.set_experiment(exp_name)

    run_name = data.run_name
    log.info(f'Starting run "{run_name}" for experiment "{exp_name}"')
    device = train_params.device
    with mlflow.start_run(run_name=run_name):
        exp_id = mlflow.active_run().info.experiment_id
        run_id = mlflow.active_run().info.run_id

        metric_results, dict_error_analysis = predict(predictor, test, device)

        mlflow.log_metrics({f'val_{name.lower()}': metric_val for name, metric_val in metric_results})
        with ArtifactWriter(job_id, run_id) as w:
            w.save_data_cfg(dl_cfg.model_dump())
            w.save_meta(meta.to_dict())
            w.save_model_cfg(model_cfg.model_dump())
            w.save_model_state(predictor.get_model().state_dict())
            w.save_train_cfg(train_cfg.model_dump())
            w.save_error_analysis(dict_error_analysis)
            w.save_post_processor_cfg(pp_cfg.model_dump())
            w.log_artifacts()

        parent_url = f'{mlflow_public_uri}/#/experiments/{exp_id}/runs/{parent_id}'
        mlflow.set_tag('Parent_run_id', parent_url)
    mlflow.end_run()
    result = ''
    ctx_dict = {}
    log.info('Training model process is successfully finished')
    return result, ctx_dict