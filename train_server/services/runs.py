from train_server.app_ctx import AppContext

from packages.server_lib.runs.run_ctx import RunContext, create_run_ctx

from train_server.schemas.runs import NewRunCfg

from packages.logger.logger import get_logger

log = get_logger(__name__)

async def create_run(ctx: AppContext, data: NewRunCfg) -> RunContext:
    run_ctx = create_run_ctx(data)
    async with ctx.runs_lock:
        ctx.runs[run_ctx.run_id] = run_ctx
    return run_ctx

async def get_run(ctx: AppContext, run_id: str) -> RunContext:
    async with ctx.runs_lock:
        run = ctx.runs[run_id]
        if run is None:
            raise RuntimeError(f'Run with id: ({run_id}) does not exist in the app_ctx')
        return run

# async def update_job(ctx: AppContext, job_id, **kwargs) -> bool:
#     async with ctx.run_lock:
#         job = ctx.runs[job_id]
#         if not job:
#             log.error(f'There is no job for updating with id: {job_id}')
#             return False
#         ctx.runs[job_id] = job.model_copy(update=kwargs)
#         return True