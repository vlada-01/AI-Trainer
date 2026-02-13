import traceback
from fastapi import APIRouter, Request, HTTPException

from app_src.schemas.runs import NewRunCfg, RunCtxResponse, ErrorInfo
from app_src.services.runs import create_run, get_run

from app_src.routes.runs_routes.jobs import router as jobs_router

from common.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])

router.include_router(jobs_router)

@router.post('/', response_model=RunCtxResponse)
async def new_run(request: Request, data: NewRunCfg):
    try:
        ctx = request.state.ctx
        run = await create_run(ctx)
        kwargs = await run.get_info()
        return RunCtxResponse(**kwargs)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc()
            )
        )

@router.get('/{run_id}', response_model=RunCtxResponse)
async def get_current_status(request: Request, run_id: str):
    try:
        ctx = request.state.ctx
        run = await get_run(ctx, run_id)
        kwargs = await run.get_info()
        return await RunCtxResponse(**kwargs)
    except Exception as e:
        log.error(f'{type(e).__name__}, {str(e)}')
        raise  HTTPException(
            status_code=500,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc()
            )
        )

# TODO: implement client cancel
# @router.post('/{run_id}/cancel', response_model=RunCtxResponse)
# async def cancel_run(request: Request, data: CancelRunCfg):
#     pass