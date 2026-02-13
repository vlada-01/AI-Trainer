from fastapi import APIRouter, Request, HTTPException
import asyncio

import app_src.schemas.job_request as requests
from app_src.schemas.job_response import JobResponse

from app_src.utils.data import atomic_prepare_dataset
from app_src.utils.models import atomic_prepare_predictor
from app_src.utils.train import atomic_train_model
from app_src.utils.predict import atomic_predict
from app_src.utils.train import atomic_inspect_run
from app_src.utils.train import atomic_post_process

from app_src.services.jobs import create_job, get_job, start_job

from app_src.services.runs import get_run

from common.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/{run_id}/jobs", tags=["jobs"])

# TODO: Need to test All transformation types
@router.post('/prepare-dataset', response_model=JobResponse)
async def prepare_dataset(request: Request, run_id: str, data: requests.DatasetJobRequest):
    log.info(f'Requesting data preparation')
    ctx = request.app.state.ctx
    run = await get_run(ctx, run_id)
    job = await create_job(run)
    params = (data)
    fn = atomic_prepare_dataset
    asyncio.create_task(start_job(run, job.id, fn, params))
    return job

# TODO: need to add check for input/output size connections
# TODO: need to check for other layer types
@router.post('/prepare-model', response_model=JobResponse)
async def prepare_model(request: Request, run_id: str, data: requests.ModelJobRequest):
    log.info('Requesting model preparation')
    ctx = request.app.state.ctx
    run = await get_run(ctx, run_id)
    job = await create_job(run)
    params = (data)
    fn = atomic_prepare_predictor
    asyncio.create_task(start_job(run, job.id, fn, params))
    return job

@router.post('/train-model', response_model=JobResponse)
async def train_model(request: Request, run_id: str, data: requests.TrainJobRequest):
    log.info('Requesting train model')
    ctx = request.app.state.ctx
    run = await get_run(ctx, run_id)
    job = await create_job(run)
    params = run.get_train_params() + (data,)
    fn = atomic_train_model
    asyncio.create_task(start_job(run, job.id, fn, params))
    return job

# TODO: add endpoint for client that has complete cfg,

@router.post('/inspect-run', response_model=JobResponse)
async def inspect_run(request: Request, run_id: str, data: requests.InspectRunJobRequest):
    ctx = request.app.state.ctx
    run = await get_run(ctx, run_id)
    job = await create_job(run)
    params = (ctx.mlflow_client, data)
    fn = atomic_inspect_run
    asyncio.create_task(atomic_inspect_run(ctx, job.id, fn, params))
    return job

# TODO: update me
@router.post('/post-process-run', response_model=JobResponse)
async def post_process(request: Request, run_id: str, data: requests.PostProcessingJobRequest):
    ctx = request.app.state.ctx
    run = await get_run(ctx, run_id)
    job = await create_job(run)
    # params = 
    asyncio.create_task(atomic_post_process(ctx, job.id, data))
    return job

# TODO: update me
@router.post('/fine_tune_run', response_model=JobResponse)
async def fine_tune(request: Request, run_id: str, data: requests.FineTuneJobRequest):
    ctx = request.app.state.ctx
    job = await create_job(ctx)

    # asyncio.create_task(run_fine_tune(ctx, job.id, data))
    return job


# TODO: do I need this
@router.post('/final-evaluation', response_model=JobResponse)
async def final_evaluation(request: Request, run_id: str, data: requests.PredictJobRequest):
    ctx = request.app.state.ctx
    run = await get_run(ctx, run_id)
    job = await create_job(run)
    params = (data)
    fn = atomic_predict
    asyncio.create_task(start_job(run, job.id, fn, params))
    return job

@router.get('/{job_id}', response_model=JobResponse)
async def job_status(request: Request, run_id: str, job_id: str):
    ctx = request.app.state.ctx
    run = await get_run(ctx, run_id)
    job = await get_job(run, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"error": "job_not_found"})
    return job

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

