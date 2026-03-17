import os
import asyncio

from packages.server_lib.runs import Job, JobStatus
from packages.server_lib.runs import StateCode
from packages.server_lib.runs import RunContext

from train_server.schemas import ErrorInfo

jobs_ttl = int(os.getenv("JOBS_TTL", "7200"))

from packages.logger import get_logger

log = get_logger(__name__)

async def try_create_job(ctx: RunContext, state_code: StateCode) -> Job:
    if not await ctx.is_valid_to_add(state_code): 
        raise RuntimeError(f'Cannot add job when run_ctx is in state: {ctx.state}')
    
    job = Job(state_code)
    await ctx.add_job(job)
    return job

async def start_job(ctx: RunContext, job_id: str, task_fn, params) -> None:
    await ctx.update_job(job_id, status=JobStatus.in_progress)
    try:
        result, ctx_dict = await asyncio.to_thread(task_fn, *params)
        await ctx.update(ctx_dict)

        await ctx.move_state(job_id)
        await ctx.update_job(job_id, status=JobStatus.success, status_details=result)
    except Exception as e:
        await ctx.update_job(
            job_id,
            status=JobStatus.failed,
            status_details={
                'error': ErrorInfo(
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
            }
        )

async def get_job(ctx: RunContext, job_id: str) -> Job:
    return await ctx.get_job(job_id)


