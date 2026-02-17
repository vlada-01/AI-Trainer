import traceback
from fastapi import APIRouter, Request, HTTPException

from app_src.schemas.runs import NewRunCfg, RunCtxResponse, ErrorInfo
from app_src.services.runs.runs import create_run, get_run

from app_src.routes.runs_routes.exec_jobs import router as exec_router
from app_src.routes.runs_routes.prepare_jobs import router as prepare_router

from common.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])

router.include_router(exec_router)
router.include_router(prepare_router)

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
    
# add endpoint for moving run with validation everything is done for that type

# TODO: implement client cancel
# @router.post('/{run_id}/cancel', response_model=RunCtxResponse)
# async def cancel_run(request: Request, data: CancelRunCfg):
#     pass