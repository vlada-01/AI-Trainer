from fastapi import APIRouter, Request, HTTPException
import asyncio

from app_src.schemas.train import TrainJobRequest, InspectRunJobRequest, PostProcessingJobRequest, FineTuneJobRequest, JobResponse

from app_src.services.train import create_job, get_job, run_train_model, run_inspect_run, run_post_process, run_fine_tune
from common.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/train", tags=["train_models"])

@router.post('/train_model', response_model=JobResponse)
async def run_train(request: Request, data: TrainJobRequest):
    ctx = request.app.state.ctx
    job = await create_job(ctx)

    asyncio.create_task(run_train_model(ctx, job.id, data))
    return job

@router.get("/train_status/{job_id}")
async def train_status(request: Request, job_id: str):
    ctx = request.app.state.ctx

    job = await get_job(ctx, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"error": "job_not_found"})

    return job

@router.post('/inspect_run', response_model=JobResponse)
async def inspect_run(request: Request, data: InspectRunJobRequest):
    ctx = request.app.state.ctx
    job = await create_job(ctx)

    asyncio.create_task(run_inspect_run(ctx, job.id, data))
    return job

@router.get("/inspect_run/{job_id}")
async def inspect_run_status(request: Request, job_id: str):
    ctx = request.app.state.ctx

    job = await get_job(ctx, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"error": "job_not_found"})

    return job

@router.post('/post_process_run', response_model=JobResponse)
async def post_process(request: Request, data: PostProcessingJobRequest):
    ctx = request.app.state.ctx
    job = await create_job(ctx)

    asyncio.create_task(run_post_process(ctx, job.id, data))
    return job

@router.get("/post_process_run/{job_id}")
async def post_process_status(request: Request, job_id: str):
    ctx = request.app.state.ctx

    job = await get_job(ctx, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"error": "job_not_found"})

    return job

@router.post('/fine_tune_run', response_model=JobResponse)
async def fine_tune(request: Request, data: FineTuneJobRequest):
    ctx = request.app.state.ctx
    job = await create_job(ctx)

    asyncio.create_task(run_fine_tune(ctx, job.id, data))
    return job

@router.get("/fine_tune_status/{job_id}")
async def fine_tune_status(request: Request, job_id: str):
    ctx = request.app.state.ctx

    job = await get_job(ctx, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"error": "job_not_found"})

    return job