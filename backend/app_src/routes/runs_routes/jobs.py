from fastapi import APIRouter, Request, HTTPException
import asyncio

import app_src.schemas.job_request as requests
from app_src.schemas.job_response import JobResponse, ErrorInfo

import app_src.services.jobs.tasks.prepare as prepare
from app_src.services.jobs.tasks.train import atomic_train_model
from app_src.services.jobs.tasks.final_evaluation import atomic_predict

from backend.app_src.services.jobs.jobs import create_job, get_job, start_job

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
        job = await create_job(run, 'PrepareDatasetJobRequest')
        params = (data)
        fn = prepare.atomic_prepare_dataset
        asyncio.create_task(start_job(run, job.id, fn, params))
        return job
    except Exception as e:
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
        job = await create_job(run, 'PrepareModelJobRequest')
        params = (data)
        fn = prepare.atomic_prepare_predictor
        asyncio.create_task(start_job(run, job.id, fn, params))
        return job
    except Exception as e:
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
        job = await create_job(run, 'PrepareTrainJobRequest')
        # TODO: need to prevent create_job if model or data is not loaded
        params = await run.get_prepare_train_params() + (data, )
        fn = prepare.atomic_prepare_train_params
        asyncio.create_task(start_job(run, job.id, fn, params))
        return job
    except Exception as e:
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
        job = await create_job(run, 'PrepareCompleteTrainJobRequest')
        params = (data)
        fn = prepare.atomic_prepare_complete_train
        asyncio.create_task(start_job(run, job.id, fn, params))
        return job
    except Exception as e:
        raise  HTTPException(
            status_code=500,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )

@router.post('/train', response_model=JobResponse)
async def train_model(request: Request, run_id: str, data: requests.StartTrainJobRequest):
    try:
        log.info('Requesting train model')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await create_job(run, 'StartTrainJobRequest')
        # TODO: need to prevent create_job if not data is loaded
        params = await run.get_train_params() + (job.id, )
        fn = atomic_train_model
        asyncio.create_task(start_job(run, job.id, fn, params))
        return job
    except Exception as e:
            raise  HTTPException(
                status_code=500,
                detail=ErrorInfo(
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
            )

@router.post('/post-process-run', response_model=JobResponse)
async def post_process(request: Request, run_id: str, data: requests.PreparePostProcessingJobRequest):
    try:
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await create_job(run, 'PreparePostProcessingJobRequest')
        params = (data)
        fn = atomic_post_process
        asyncio.create_task(start_job(ctx, job.id, fn, params))
        return job
    except Exception as e:
        raise  HTTPException(
            status_code=500,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )

@router.post('/fine_tune_run', response_model=JobResponse)
async def fine_tune(request: Request, run_id: str, data: requests.FineTuneJobRequest):
    try:
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await create_job(run, 'FineTuneJobRequest')
        params = (data)
        fn = atomic_fine_tune
        asyncio.create_task(start_job(ctx, job.id, fn, params))
        return job
    except Exception as e:
        raise  HTTPException(
            status_code=500,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )

@router.post('/final-evaluation', response_model=JobResponse)
async def final_evaluation(request: Request, run_id: str, data: requests.PredictJobRequest):
    try:
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await create_job(run, 'PredictJobRequest')
        params = (data)
        fn = atomic_predict
        asyncio.create_task(start_job(run, job.id, fn, params))
        return job
    except Exception as e:
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

