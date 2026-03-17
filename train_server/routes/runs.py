import traceback
from fastapi import APIRouter, Request, HTTPException

from train_server.schemas import NewRunCfg, RunCtxResponse, ErrorInfo, JobResponse
from train_server.services.runs import create_run, get_run
from train_server.services.jobs.jobs import get_job

from .runs_routes import exec_jobs_router, prepare_jobs_router

from packages.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])

router.include_router(exec_jobs_router)
router.include_router(prepare_jobs_router)

@router.post('/', response_model=RunCtxResponse)
async def new_run(request: Request, data: NewRunCfg):
    try:
        log.info('Requesting new run initialization')
        ctx = request.app.state.ctx
        run = await create_run(ctx, data)
        kwargs = await run.get_info()
        log.info('New Run is successfully prepared')
        return RunCtxResponse(**kwargs)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc().splitlines()
            )
        )

@router.get('/{run_id}', response_model=RunCtxResponse)
async def get_current_status(request: Request, run_id: str):
    try:
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        kwargs = await run.get_info()
        return RunCtxResponse(**kwargs)
    except Exception as e:
        print(traceback.format_exc())
        raise  HTTPException(
            status_code=500,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc().splitlines()
            )
        )
    
@router.get('/{run_id}/{job_id}', response_model=JobResponse)
async def job_status(request: Request, run_id: str, job_id: str):
    try:
        ctx = request.app.state.ctx
        run = await get_run(ctx, run_id)
        job = await get_job(run, job_id)
        return JobResponse(**job.to_dict())
    except Exception as e:
        log.critical(traceback.format_exc())
        raise  HTTPException(
            status_code=404,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )

    
# add endpoint for moving run with validation everything is done for that type

# TODO: implement client cancel
# @router.post('/{run_id}/cancel', response_model=RunCtxResponse)
# async def cancel_run(request: Request, data: CancelRunCfg):
#     pass