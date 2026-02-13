from fastapi import APIRouter, Request

from app_src.schemas.runs import NewRunCfg

from app_src.services.runs import create_run, get_run

from app_src.routes.runs_routes.jobs import router as jobs_router

from common.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])

router.include_router(jobs_router)
# TODO: might need to add exceptions for these endpoints

@router.post('/')
async def new_run(request: Request, data: NewRunCfg):
    ctx = request.state.ctx
    run = await create_run(ctx)
    return run.get_info()

@router.get('/{run_id}')
async def get_current_status(request: Request, run_id: str):
    ctx = request.state.ctx
    run = await get_run(ctx, run_id)
    return run.get_info()