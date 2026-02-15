import mlflow
import copy

from model_src.models.post_processor import build_post_processor

from model_src.eval import evaluate

from app_src.app import mlflow_public_uri

from app_src.services.reader_writer import ArtifactWriter

from common.logger import get_logger

log = get_logger(__name__)

def atomic_post_process(predictor, val_dl, train_params, meta, dl_cfg, model_cfg, train_cfg, parent_id, pp_cfg, job_id):
    log.info('Initiazling post processsor process')
    if parent_id is None:
        raise RuntimeError('Cannot initialize post processing wihtout parent mlflow run id')
    
    log.info('Initializing post processor')
    pp = build_post_processor(pp_cfg)

    log.info('Initializing training of post processor parameters')
    device = train_params.device
    pp.train(copy.deepcopy(predictor.get_model()), val_dl, device)
    
    log.info('Setting post processor in the predictor')
    predictor.set_pp(pp)

    updated_pp_cfg = predictor.get_pp().get_cfg()
    
    exp_name = train_params.train_cfg.exp_name
    mlflow.set_experiment(exp_name)
    log.info(f'Starting run "{pp_cfg.new_run_name}" for experiment "{exp_name}"')
    with mlflow.start_run(run_name=pp_cfg.new_run_name):
        exp_id = mlflow.active_run().info.experiment_id
        run_id = mlflow.active_run().info.run_id
        val_metrics, dict_error_analysis = evaluate(predictor, val_dl, train_params, collect_error_analysis=True)

        mlflow.log_metrics({f'val_{name.lower()}': metric_val for name, metric_val in val_metrics})
        with ArtifactWriter(job_id, run_id) as w:
            w.save_data_cfg(dl_cfg.model_dump())
            w.save_meta(meta.to_dict())
            w.save_model_cfg(model_cfg.model_dump())
            w.save_model_state(predictor.get_model().state_dict())
            w.save_train_cfg(train_cfg.model_dump())
            w.ssave_error_analysis(dict_error_analysis)
            w.save_post_processor_cfg(updated_pp_cfg.model_dump())
            w.log_artifacts()

        parent_url = f'{mlflow_public_uri}/#/experiments/{exp_id}/runs/{parent_id}'
        mlflow.set_tag('Parent_run_id', parent_url)
    mlflow.end_run()

    # TODO: why I have this in legacy
    # result['layer_cfg'] = model_cfg.model_dump(exclude={'model_type'})

    result = ''
    ctx_dict = {}
    log.info('Post processor is successfully finished')
    return result, ctx_dict