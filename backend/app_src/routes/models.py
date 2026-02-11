from fastapi import APIRouter, Request, HTTPException
import asyncio

from app_src.schemas.models import ModelJobRequest, ModelJobResponse

from app_src.services.models import create_job, get_job, run_prepare_model
from common.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["models"])

# TODO: need to add check for input/output size connections
# TODO: need to check for other layer types
@router.post('/prepare_model', response_model=ModelJobResponse)
async def prepare_model(request: Request, data: ModelJobRequest):
    log.info(f'Requesting model preparation')
    ctx = request.app.state.ctx
    job = await create_job(ctx)

    asyncio.create_task(run_prepare_model(ctx, job.id, data))
    return job

@router.get("/model_status/{job_id}")
async def model_status(request: Request, job_id: str):
    ctx = request.app.state.ctx

    job = await get_job(ctx, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"error": "job_not_found"})

    return job

