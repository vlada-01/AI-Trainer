from app_src.app_ctx import AppContext

from app_src.app_ctx import RunContext

from common.logger import get_logger

log = get_logger(__name__)

async def create_run(ctx: AppContext) -> RunContext:
    run = RunContext()

    async with ctx.runs_lock:
        ctx.runs[run.run_id] = run

    return run

async def get_run(ctx: AppContext, run_id: str) -> RunContext | None:
    async with ctx.runs_lock:
        run = ctx.runs[run_id]
        return run if run is not None else None
    
async def update_job(ctx: AppContext, job_id, **kwargs) -> bool:
    async with ctx.jobs_lock:
        job = ctx.jobs[job_id]
        if not job:
            log.error(f'There is no job for updating with id: {job_id}')
            return False
        ctx.jobs[job_id] = job.model_copy(update=kwargs)
        return True