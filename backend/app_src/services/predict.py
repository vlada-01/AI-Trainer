import asyncio
import traceback
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from app_src.app_ctx import AppContext
from app_src.schemas.predict import PredictJobRequest, JobResponse, ErrorInfo

from app_src.utils.predict import predict

from common.logger import get_logger

log = get_logger(__name__)

JOB_TTL_SECONDS = 60 * 30

async def create_job(ctx: AppContext) -> JobResponse:
    job_id = uuid4().hex
    job = JobResponse(
        id=job_id,
        status='pending',
        created_at=datetime.now(timezone.utc).isoformat(),
        expires_at=(datetime.now(timezone.utc) + timedelta(seconds=JOB_TTL_SECONDS)).isoformat()
    )

    async with ctx.jobs_lock:
        ctx.jobs[job_id] = job

    return job

async def get_job(ctx: AppContext, job_id: str) -> JobResponse | None:
    async with ctx.jobs_lock:
        job = ctx.jobs[job_id]
        return job if job is not None else None
    
async def run_predict(ctx: AppContext, job_id: str, cfg: PredictJobRequest) -> None:
    await update_job(ctx, job_id, status='in_progress')
    try:
        params = await get_predict_params(ctx, cfg)
        result = await asyncio.to_thread(predict, *params)
        async with ctx.jobs_lock:
            ctx.predictor = None
            ctx.val = None
            ctx.test = None
            ctx.meta = None
            ctx.cached_model_cfg = None
            ctx.cached_dl_cfg = None
            ctx.cached_train_cfg = None

        await update_job(
            ctx,
            job_id,
            status="success",
            status_details=result
        )
    except Exception as e:
        await update_job(
            ctx,
            job_id,
            status="failed",
            status_details="Predicting the dataset failed",
            error=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc().splitlines()
            )
        )

async def update_job(ctx: AppContext, job_id, **kwargs) -> bool:
    async with ctx.jobs_lock:
        job = ctx.jobs[job_id]
        if not job:
            log.error(f'There is no job for updating with id: {job_id}')
            return False
        ctx.jobs[job_id] = job.model_copy(update=kwargs)
        return True
    
async def get_predict_params(ctx: AppContext, cfg: PredictJobRequest):
    async with ctx.jobs_lock:
        client = ctx.mlflow_client
        return client, cfg