import asyncio
import traceback
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from app_src.app_ctx import AppContext
from app_src.schemas.models import ModelJobRequest, ModelJobResponse, ErrorInfo

from app_src.utils.models import prepare_model

from common.logger import get_logger

log = get_logger(__name__)

JOB_TTL_SECONDS = 60 * 30

async def create_job(ctx: AppContext) -> ModelJobResponse:
    job_id = uuid4().hex
    job = ModelJobResponse(
        id=job_id,
        status='pending',
        created_at=datetime.now(timezone.utc).isoformat(),
        expires_at=(datetime.now(timezone.utc) + timedelta(seconds=JOB_TTL_SECONDS)).isoformat()
    )

    async with ctx.jobs_lock:
        ctx.jobs[job_id] = job

    return job

async def get_job(ctx: AppContext, job_id: str) -> ModelJobResponse | None:
    async with ctx.jobs_lock:
        job = ctx.jobs[job_id]
        return job if job is not None else None
    
async def run_prepare_model(ctx: AppContext, job_id: str, cfg: ModelJobRequest) -> None:
    await update_job(ctx, job_id, status='in_progress')
    try:
        result = await asyncio.to_thread(prepare_model, cfg)
        async with ctx.jobs_lock:
            ctx.model = result['model']
            ctx.cached_model_cfg = result['cfg']

        await update_job(
            ctx,
            job_id,
            status="success",
            status_details="Model prepared successfully",
        )
    except Exception as e:
        await update_job(
            ctx,
            job_id,
            status="failed",
            status_details="Model preparation failed",
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
            log.warning(f'There is no job for updating with id: {job_id}')
            return False
        ctx.jobs[job_id] = job.model_copy(update=kwargs)
        return True