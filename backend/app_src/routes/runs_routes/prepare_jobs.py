from fastapi import APIRouter, Request, HTTPException
import asyncio
import traceback

import app_src.schemas.job_request as requests
from app_src.schemas.job_response import JobResponse, ErrorInfo

from app_src.services.runs.state_manager import StateCode

import app_src.services.jobs.tasks.prepare_tasks as prepare

from app_src.services.jobs.jobs import try_create_job, get_job, start_job

from app_src.services.runs.runs import get_run

from common.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/{run_id}/prepare-jobs", tags=["jobs"])

# SKLEARN is deprecated, might be deleted
@router.post('/dataset', response_model=JobResponse)
async def prepare_dataset(request: Request, run_id: str, data: requests.PrepareDatasetJobRequest):
    try:
        log.info(f'Requesting data preparation')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.prepare_ds)
        params = (data, )
        fn = prepare.atomic_prepare_dataset
        asyncio.create_task(start_job(run, job.id, fn, params))
        return job
    except Exception as e:
        print(traceback.format_exc())
        raise  HTTPException(
            status_code=500,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )

@router.post('/model', response_model=JobResponse)
async def prepare_model(request: Request, run_id: str, data: requests.PrepareModelJobRequest):
    try:
        log.info('Requesting model preparation')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.prepare_model)
        params = (data, ) #written like this, seems like there is no dependency from ds
        fn = prepare.atomic_prepare_predictor
        asyncio.create_task(start_job(run, job.id, fn, params))
        return job
    except Exception as e:
        print(traceback.format_exc())
        raise  HTTPException(
            status_code=500,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )
    
@router.post('/train-params', response_model=JobResponse)
async def prepare_train(request: Request, run_id: str, data: requests.PrepareTrainJobRequest):
    try:
        log.info('Requesting train parameters preparation')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.prepare_default)
        # TODO: need to prevent create_job if model or data is not loaded
        params = await run.get_prepare_train_params() + (data, )
        fn = prepare.atomic_prepare_train_params
        asyncio.create_task(start_job(run, job.id, fn, params))
        return job
    except Exception as e:
        print(traceback.format_exc())
        raise  HTTPException(
            status_code=500,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )

@router.post('/full-train', response_model=JobResponse)
async def prepare_train(request: Request, run_id: str, data: requests.PrepareCompleteTrainJobRequest):
    try:
        log.info('Requesting complete train preparation')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.prepare_default)
        params = (data, )
        fn = prepare.atomic_prepare_complete_train
        asyncio.create_task(start_job(run, job.id, fn, params))
        return job
    except Exception as e:
        print(traceback.format_exc())
        raise  HTTPException(
            status_code=500,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )
    
@router.post('/load-run', response_model=JobResponse)
async def load_run_cfg(request: Request, run_id: str, data: requests.LoadRunCfgJobRequest):
    try:
        log.info('Requesting complete train preparation')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.prepare_default_run)
        params = (data, job.id)
        fn = prepare.atomic_prepare_default_from_run
        asyncio.create_task(start_job(run, job.id, fn, params))
        return job
    except Exception as e:
        print(traceback.format_exc())
        raise  HTTPException(
            status_code=500,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )

# TODO update later to be able to repeatedly add more and more post processings on top of same base
@router.post('/post-process', response_model=JobResponse)
async def post_process(request: Request, run_id: str, data: requests.PreparePostProcessingJobRequest):
    try:
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.prepare_pp)
        params = await run.get_post_process_params() + (data, )
        fn = prepare.atomic_prepare_post_process
        asyncio.create_task(start_job(ctx, job.id, fn, params))
        return job
    except Exception as e:
        print(traceback.format_exc())
        raise  HTTPException(
            status_code=500,
            detail=ErrorInfo(
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
    
@router.get('/{job_id}', response_model=JobResponse)
async def job_status(request: Request, run_id: str, job_id: str):
    try:
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await get_job(run, job_id)
        return job
    except Exception as e:
        print(traceback.format_exc())
        raise  HTTPException(
            status_code=404,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )
