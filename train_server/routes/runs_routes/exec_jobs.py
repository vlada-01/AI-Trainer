from fastapi import APIRouter, Request, HTTPException
import asyncio
import traceback

import train_server.schemas as schemas
from train_server.services.jobs.tasks.train import atomic_train_model
from train_server.services.jobs.tasks.final_evaluation import atomic_final_eval
from train_server.services.jobs.jobs import try_create_job, get_job, start_job
from train_server.services.runs import get_run

from packages.server_lib.runs import StateCode

from packages.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/{run_id}/exec-jobs", tags=["jobs"])

#TODO: add support for streamed data training
@router.post('/train', response_model=schemas.JobResponse)
async def train_model(request: Request, run_id: str, data: schemas.StartTrainJobRequest):
    try:
        log.info('Requesting train model')
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.training)
        params = await run.get_train_params() + (data, job.id)
        fn = atomic_train_model
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

@router.post('/final-evaluation', response_model=schemas.JobResponse)
async def final_evaluation(request: Request, run_id: str, data: schemas.FinalEvalJobRequest):
    try:
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await try_create_job(run, StateCode.final_eval)
        params = await run.get_final_eval_params() + (data, job.id)
        fn = atomic_final_eval
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

