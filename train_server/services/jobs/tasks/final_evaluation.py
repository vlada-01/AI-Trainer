import os
import mlflow

from train_server.services.reader_writer import ArtifactWriter

from packages.logger.logger import get_logger

log = get_logger(__name__)

mlflow_public_uri = os.getenv("MLFLOW_PUBLIC_URI", "http://localhost:5000")

# FIXME: add more tags for the run
def atomic_final_eval(engine, parent_id, meta, cfgs, data, job_id):
    log.info('Initializing final evaluation process')
    exp_name = data.exp_name
    log.info(f'Setting experiment name: {exp_name}')
    mlflow.set_experiment(exp_name)

    # FIXME: the data.model_name is not used, but should be able to

    run_name = data.run_name
    log.info(f'Starting run "{run_name}" for experiment "{exp_name}"')
    with mlflow.start_run(run_name=run_name):
        exp_id = mlflow.active_run().info.experiment_id
        run_id = mlflow.active_run().info.run_id

        #FIXME: this crap is wrong, need to store error_analysis_dict in the file, not dict
        metrics_results, error_analysis_dict = engine.evaluate_test()
        model = engine.get_model()

        for k, metrics in metrics_results.items():
            mlflow.log_metrics({f'test/{k}/{name.lower()}': metric_val for name, metric_val in metrics})
        with ArtifactWriter(job_id, run_id) as w:
            w.save_data_cfg(cfgs.dl_cfg.model_dump())
            w.save_model_cfg(cfgs.model_cfg.model_dump())
            w.save_train_cfg(cfgs.train_cfg.model_dump())
            # FIXME: need to update meta properly
            # w.save_meta(meta.to_dict())
            w.save_model_state(model.state_dict())
            w.save_error_analysis(error_analysis_dict)
            w.log_artifacts()

        parent_url = f'{mlflow_public_uri}/#/experiments/{exp_id}/runs/{parent_id}'
        mlflow.set_tag('Parent_run_id', parent_url)
    mlflow.end_run()
    result = ''
    ctx_dict = {}
    log.info('Training model process is successfully finished')
    return result, ctx_dict