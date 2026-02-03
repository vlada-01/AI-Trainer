from fastapi import APIRouter, Request, HTTPException
import asyncio

from app_src.schemas.predict import PredictJobRequest, JobResponse

from app_src.services.predict import create_job, get_job, run_predict
from common.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["predict"])

@router.post('/test', response_model=JobResponse)
async def predict(request: Request, data: PredictJobRequest):
    ctx = request.app.state.ctx
    job = await create_job(ctx)

    asyncio.create_task(run_predict(ctx, job.id, data))
    return job

@router.get("/test_status/{job_id}")
async def predict_status(request: Request, job_id: str):
    ctx = request.app.state.ctx

    job = await get_job(ctx, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"error": "job_not_found"})

    return job