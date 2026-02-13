import traceback
from fastapi import APIRouter, Request, HTTPException

from app_src.schemas.history import HistoryResponse, Experiment
from app_src.schemas.job_response import ErrorInfo

from app_src.app import mlflow_public_uri

from common.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])

@router.get('/history', response_model=HistoryResponse)
def history(request: Request):
    try:
        ctx = request.state.ctx
        exps = ctx.mlflow_client.search_experiments()
        active_exps = [e for e in exps if e.lifecycle_stage == 'active']
        list_of_exps = [
            Experiment(
                name=e.name,
                url=f'{mlflow_public_uri}/#/experiments/{e.experiment_id}'
            )
            for e in active_exps
        ]
        return HistoryResponse(
            exps=list_of_exps
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc()
            )
        )