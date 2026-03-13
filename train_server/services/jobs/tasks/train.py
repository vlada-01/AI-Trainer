import mlflow

from train_server.services.reader_writer import ArtifactWriter

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
        mlflow.log_params(cfgs.train_cfg.model_dump())

        log.info('Initializing Model Training')
        engine.train_model()

        log.info('Initializing Post Processor train')
        engine.train_pp()

        log.info('Initializing validation set evaluation')
        # FIXME: This crap shall log the error analysis in the file and this file shall be stored, not dict
        _, error_analysis_dict = engine.evaluate_val()

        model = engine.get_model()

        run_id = mlflow.active_run().info.run_id
        with ArtifactWriter(job_id, run_id) as w:
            w.save_data_cfg(cfgs.dl_cfg.model_dump())
            w.save_model_cfg(cfgs.model_cfg.model_dump())
            w.save_train_cfg(cfgs.train_cfg.model_dump())
            # FIXME: need to store new meta properly
            # w.save_meta(meta.to_dict())
            w.save_model_state(model.state_dict())
            w.save_error_analysis(error_analysis_dict)
            w.log_artifacts()

    mlflow.end_run()
    result = 'Training is finished successfully'
    ctx_dict = {}
    log.info('Training model process is successfully finished')
    return result, ctx_dict