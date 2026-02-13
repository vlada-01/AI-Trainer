import asyncio
import traceback
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from app_src.run_ctx import RunContext
from app_src.schemas.job_response import JobResponse, ErrorInfo

from common.logger import get_logger

log = get_logger(__name__)

JOB_TTL_SECONDS = 60 * 30

async def create_job(ctx: RunContext) -> JobResponse:
    job_id = uuid4().hex
    job = JobResponse(
        id=job_id,
        status='pending',
        created_at=datetime.now(timezone.utc).isoformat(),
        expires_at=(datetime.now(timezone.utc) + timedelta(seconds=JOB_TTL_SECONDS)).isoformat()
    )

    async with ctx.jobs_lock:
        ctx.jobs[job.id] = job

    return job

async def get_job(ctx: RunContext, job_id: str) -> JobResponse | None:
    async with ctx.jobs_lock:
        job = ctx.jobs[job_id]
        return job if job is not None else None
    
async def update_job(ctx: RunContext, job_id, **kwargs) -> bool:
    async with ctx.jobs_lock:
        job = ctx.jobs[job_id]
        if not job:
            log.error(f'There is no job for updating with id: {job_id}')
            return False
        ctx.jobs[job_id] = job.model_copy(update=kwargs)
        return True

async def start_job(ctx: RunContext, job_id: str, fn, params) -> None:
    await update_job(ctx, job_id, status='in_progress')
    try:
        # TODO: make functions return result, for update_job and ctx_dict for updating the run_ctx
        result = await asyncio.to_thread(fn, *params)
        async with ctx.jobs_lock:
            ctx.update(result)

        await update_job(
            ctx,
            job_id,
            status="success",
            # update this return message as well
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