import asyncio
import traceback
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from app_src.services.runs.run_ctx import RunContext
from app_src.schemas.job_response import JobResponse, ErrorInfo

from app_src.app import jobs_ttl

from common.logger import get_logger

log = get_logger(__name__)

async def create_job(ctx: RunContext, job_type: str) -> JobResponse:
    # TODO: change this to get statusCode
    if await ctx.is_valid_to_add():
        raise RuntimeError(f'Cannot add job when run_ctx is in state: {ctx.status}')
    job_id = uuid4().hex
    job = JobResponse(
        id=job_id,
        job_type = job_type,
        status='pending',
        created_at=datetime.now(timezone.utc).isoformat(),
        expires_at=(datetime.now(timezone.utc) + timedelta(seconds=jobs_ttl)).isoformat()
    )

    async with ctx.run_ctx_lock:
        ctx.jobs[job.id] = job

    return job

async def start_job(ctx: RunContext, job_id: str, task_fn, params) -> None:
    await update_job(ctx, job_id, status='in_progress')
    try:
        result, ctx_dict = await asyncio.to_thread(task_fn, *params)
        await ctx.update(ctx_dict)

        #TODO: update statecode
        await update_job(
            ctx,
            job_id,
            status="success",
            status_details=result
        )
    except Exception as e:
        # TODO: finishes job but, does not send to client that it failed, until client requests it again
        # update statecode
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

async def get_job(ctx: RunContext, job_id: str) -> JobResponse:
    async with ctx.run_ctx_lock:
        job = ctx.jobs[job_id]
        if job is None:
            raise RuntimeError(f'Job with id: ({job_id}) does not exist in the run_ctx with id: ({ctx.run_id})')
        return job
    
async def update_job(ctx: RunContext, job_id, **kwargs) -> bool:
    async with ctx.run_ctx_lock:
        job = ctx.jobs[job_id]
        if not job:
            log.error(f'There is no job for updating with id: {job_id}')
            return False
        ctx.jobs[job_id] = job.model_copy(update=kwargs)
        return True

