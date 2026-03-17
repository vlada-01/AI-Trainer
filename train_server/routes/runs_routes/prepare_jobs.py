from fastapi import APIRouter, Request, HTTPException
import asyncio
import traceback

import train_server.schemas as schemas
import train_server.services.jobs.tasks.prepare_tasks as prepare
from train_server.services.jobs.jobs import try_create_job, get_job, start_job
from train_server.services.runs import get_run

from packages.server_lib.runs import StateCode

from packages.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/{run_id}/prepare-jobs", tags=["jobs"])

@router.post('/dataset', response_model=schemas.JobResponse)
async def prepare_dataset(request: Request, run_id: str, data: schemas.PrepareDatasetJobRequest):
    try:
        log.info(f'Requesting data preparation')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.prepare_ds)
        params = (data, )
        fn = prepare.atomic_prepare_dataset
        asyncio.create_task(start_job(run, job.id, fn, params))
        return schemas.JobResponse(**job.to_dict())
    except Exception as e:
        log.critical(traceback.format_exc())
        raise  HTTPException(
            status_code=500,
            detail=schemas.ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )

@router.post('/model', response_model=schemas.JobResponse)
async def prepare_model(request: Request, run_id: str, data: schemas.PrepareModelJobRequest):
    try:
        log.info('Requesting model preparation')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.prepare_model)
        params = await run.get_prepare_model_params() + (data, ) #written like this, seems like there is no dependency from ds
        fn = prepare.atomic_prepare_model
        asyncio.create_task(start_job(run, job.id, fn, params))
        return schemas.JobResponse(**job.to_dict())
    except Exception as e:
        log.critical(traceback.format_exc())
        raise  HTTPException(
            status_code=500,
            detail=schemas.ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )
    
@router.post('/engine', response_model=schemas.JobResponse)
async def prepare_engine(request: Request, run_id: str, data: schemas.PrepareTrainJobRequest):
    try:
        log.info('Requesting train parameters preparation')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.prepare_default)
        params = await run.get_prepare_engine_params() + (data, )
        fn = prepare.atomic_prepare_engine
        asyncio.create_task(start_job(run, job.id, fn, params))
        return schemas.JobResponse(**job.to_dict())
    except Exception as e:
        log.critical(traceback.format_exc())
        raise  HTTPException(
            status_code=500,
            detail=schemas.ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )

@router.post('/full-train', response_model=schemas.JobResponse)
async def prepare_copmplete_train(request: Request, run_id: str, data: schemas.PrepareCompleteTrainJobRequest):
    try:
        log.info('Requesting complete train preparation')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.prepare_default)
        params = await run.get_prepare_complete_train_params() + (data, )
        fn = prepare.atomic_prepare_complete_train
        asyncio.create_task(start_job(run, job.id, fn, params))
        return schemas.JobResponse(**job.to_dict())
    except Exception as e:
        log.critical(traceback.format_exc())
        raise  HTTPException(
            status_code=500,
            detail=schemas.ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )
    
@router.post('/load-run', response_model=schemas.JobResponse)
async def load_run_cfg(request: Request, run_id: str, data: schemas.LoadRunCfgJobRequest):
    try:
        log.info('Requesting complete train preparation')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.prepare_default_run)
        params = await run.get_prepare_default_from_run_params() + (data, job.id)
        fn = prepare.atomic_prepare_default_from_run
        asyncio.create_task(start_job(run, job.id, fn, params))
        return schemas.JobResponse(**job.to_dict())
    except Exception as e:
        log.critical(traceback.format_exc())
        raise  HTTPException(
            status_code=500,
            detail=schemas.ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )

@router.post('/post-process', response_model=schemas.JobResponse)
async def post_process(request: Request, run_id: str, data: schemas.PreparePostProcessingJobRequest):
    try:
        log.info('Requesting post processor preparation')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.prepare_pp)
        params = await run.get_post_process_params() + (data, )
        fn = prepare.atomic_prepare_post_process
        asyncio.create_task(start_job(run, job.id, fn, params))
        return schemas.JobResponse(**job.to_dict())
    except Exception as e:
        log.critical(traceback.format_exc())
        raise  HTTPException(
            status_code=500,
            detail=schemas.ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )
    
# TODO: update me
# @router.post('/prepare-fine_tune', response_model=JobResponse)
# async def fine_tune(request: Request, run_id: str, data: requests.FineTuneJobRequest):
#     try:
#         ctx = request.app.state.ctx
#         run = await get_run(ctx, run_id)
#         job = await try_create_job(run, StateCode.prepare_fine_tune)
#         params = (data)
#         fn = atomic_fine_tune
#         asyncio.create_task(start_job(ctx, job.id, fn, params))
#         return job
#     except Exception as e:
#         raise  HTTPException(
#             status_code=500,
#             detail=ErrorInfo(
#                 error_type=type(e).__name__,
#                 error_message=str(e)
#             )
#         )
    
@router.get('/{job_id}', response_model=schemas.JobResponse)
async def job_status(request: Request, run_id: str, job_id: str):
    try:
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await get_job(run, job_id)
        return schemas.JobResponse(**job.to_dict())
    except Exception as e:
        log.critical(traceback.format_exc())
        raise  HTTPException(
            status_code=404,
            detail=schemas.ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )
