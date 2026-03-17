import os
import mlflow

from packages.mlflow_logger import MlflowWriter

from packages.logger import get_logger

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
        writer = MlflowWriter(job_id, run_id)

        with writer.open_artifact_writer() as w:
            w.log_cfg(cfgs.dl_cfg.model_dump(), rel_path='cfgs/dataset_cfg.json')
            w.log_cfg(cfgs.model_cfg.model_dump(), rel_path='cfgs/model_cfg.json')
            w.log_cfg(cfgs.train_cfg.model_dump(), rel_path='cfgs/train_cfg.json')
            # FIXME: need to update meta properly
            # w.save_meta(meta.to_dict())
            model = engine.get_model()
            w.log_model_state(model.state_dict(), rel_path='model/model.pt')
            log.info('Initializing test evaluation')
            metrics_results, losses_results = engine.evaluate_test(w)
        writer.log_metrics(metrics_results, prefix='test')
        writer.log_losses(losses_results, prefix='test')

        parent_url = f'{mlflow_public_uri}/#/experiments/{exp_id}/runs/{parent_id}'
        writer.set_tags({'Parent url': parent_url})
    mlflow.end_run()
    result = ''
    ctx_dict = {}
    log.info('Training model process is successfully finished')
    return result, ctx_dict