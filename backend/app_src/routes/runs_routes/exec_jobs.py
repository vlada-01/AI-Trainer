from fastapi import APIRouter, Request, HTTPException
import asyncio
import traceback

import app_src.schemas.job_request as requests
from app_src.schemas.job_response import JobResponse, ErrorInfo

from app_src.services.runs.state_manager import StateCode

from app_src.services.jobs.tasks.train import atomic_train_model
from app_src.services.jobs.tasks.final_evaluation import atomic_final_eval

from app_src.services.jobs.jobs import try_create_job, get_job, start_job

from app_src.services.runs.runs import get_run

from common.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/{run_id}/exec_jobs", tags=["jobs"])

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

@router.post('/final-evaluation', response_model=JobResponse)
async def final_evaluation(request: Request, run_id: str, data: requests.FinalEvalJobRequest):
    try:
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.final_eval)
        params = await run.get_final_eval_params() + (data, job.id)
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

