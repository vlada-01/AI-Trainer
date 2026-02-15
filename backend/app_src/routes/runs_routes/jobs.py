from fastapi import APIRouter, Request, HTTPException
import asyncio
import traceback

import app_src.schemas.job_request as requests
from app_src.schemas.job_response import JobResponse, ErrorInfo

from app_src.services.runs.state_manager import StateCode

import app_src.services.jobs.tasks.prepare as prepare
from app_src.services.jobs.tasks.train import atomic_train_model
from app_src.services.jobs.tasks.post_process import atomic_post_process
from app_src.services.jobs.tasks.final_evaluation import atomic_final_eval

from app_src.services.jobs.jobs import try_create_job, get_job, start_job

from app_src.services.runs.runs import get_run

from common.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/{run_id}/jobs", tags=["jobs"])

@router.post('/prepare-dataset', response_model=JobResponse)
async def prepare_dataset(request: Request, run_id: str, data: requests.PrepareDatasetJobRequest):
    try:
        log.info(f'Requesting data preparation')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.prepare_ds)
        params = (data)
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

@router.post('/prepare-model', response_model=JobResponse)
async def prepare_model(request: Request, run_id: str, data: requests.PrepareModelJobRequest):
    try:
        log.info('Requesting model preparation')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.prepare_model)
        params = (data) #written like this, seems like there is no dependency from ds
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
    
@router.post('/prepare-train-params', response_model=JobResponse)
async def prepare_train(request: Request, run_id: str, data: requests.PrepareTrainJobRequest):
    try:
        log.info('Requesting train parameters preparation')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.default_cfg_ready)
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

@router.post('/prepare-train', response_model=JobResponse)
async def prepare_train(request: Request, run_id: str, data: requests.PrepareCompleteTrainJobRequest):
    try:
        log.info('Requesting complete train preparation')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.default_cfg_ready)
        params = (data)
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
    
@router.post('/load-run-cfg', response_model=JobResponse)
async def load_run_cfg(request: Request, run_id: str, data: requests.LoadRunCfgJobRequest):
    try:
        log.info('Requesting complete train preparation')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.default_cfg_ready)
        params = (data, job.id)
        fn = prepare.atomic_prepare_cfg_from_run
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
@router.post('/prepare-post-process', response_model=JobResponse)
async def post_process(request: Request, run_id: str, data: requests.PreparePostProcessingJobRequest):
    try:
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.prepare_pp)
        params = await run.get_post_process_params() + (data, job.id)
        fn = atomic_post_process
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

@router.post('/train', response_model=JobResponse)
async def train_model(request: Request, run_id: str, data: requests.StartTrainJobRequest):
    try:
        log.info('Requesting train model')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.training)
        # TODO: need to prevent create_job if not data is loaded
        params = await run.get_train_params() + (job.id, )
        fn = atomic_train_model
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

@router.get('/final-evaluation', response_model=JobResponse)
async def final_evaluation(request: Request, run_id: str):
    try:
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.final_eval)
        params = await run.get_final_eval_params()
        fn = atomic_final_eval
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

# TODO: big problem for not being able to cancel task once it is started, for now all jobs are atomic
# @router.delete('/{job_id}/cancel', response_model=JobResponse)
# async def cancel_job(request: Request, run_id: str, job_id: str):
#     ctx = request.app.state.ctx
#     run = await get_run(ctx, run_id)
#     job = await get_job(ctx, job_id)
#     # TODO: implement later stopping logic
#     # add exception for asyncio.CancelledError in start_job
#     # and function that was called in to_thread need to occasionally check cancel_flag
#     return None

