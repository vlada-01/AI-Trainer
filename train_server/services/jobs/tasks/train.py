import mlflow

from packages.mlflow_logger import MlflowWriter
from packages.train_lib.prepare_model import update_pps_cfg

from packages.logger.logger import get_logger

log = get_logger(__name__)

# FIXME: add more tags for the run
def atomic_train_model(engine, meta, cfgs, data, job_id):
    log.info('Initializing training model process')
    exp_name = data.exp_name
    log.info(f'Setting experiment name: {exp_name}')
    mlflow.set_experiment(exp_name)
    #FIXME: data.model_name is not used

    run_name = data.run_name
    log.info(f'Starting run "{run_name}" for experiment "{exp_name}"')
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        writer = MlflowWriter(job_id, run_id)
        writer.log_params(cfgs.train_cfg.model_dump())

        log.info('Initializing Model Training')
        engine.train_model(writer)

        log.info('Initializing Post Processor train')
        pps_cfg_update = engine.train_pp()
        updated_pps_cfg = update_pps_cfg(cfgs.model_cfg.pps_cfg, pps_cfg_update)
        cfgs.model_cfg.pps_cfg = updated_pps_cfg
        
        with writer.open_artifact_writer() as w:
            w.log_cfg(cfgs.dl_cfg.model_dump(), rel_path='cfgs/dataset_cfg.json')
            w.log_cfg(cfgs.model_cfg.model_dump(), rel_path='cfgs/model_cfg.json')
            w.log_cfg(cfgs.train_cfg.model_dump(), rel_path='cfgs/train_cfg.json')
            # FIXME: need to update meta properly
            # w.save_meta(meta.to_dict())
            model = engine.get_model()
            w.log_model_state(model.state_dict(), rel_path='model/model.pt')
            log.info('Initializing validation set evaluation')
            metrics_results = engine.evaluate_val(w)
        writer.log_metrics(metrics_results, 'validation')

    mlflow.end_run()
    result = 'Training is finished successfully'
    ctx_dict = {}
    log.info('Training model process is successfully finished')
    return result, ctx_dict