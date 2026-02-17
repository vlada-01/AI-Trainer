import traceback
from fastapi import APIRouter, Request, HTTPException

from app_src.schemas.mlflow import HistoryResponse, ResultsResponse
from app_src.schemas.job_response import ErrorInfo

from app_src.services.mlflow import get_experiments, get_run_results

from common.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/mlflow", tags=["mlflow"])

@router.get('/history', response_model=HistoryResponse)
def history(request: Request):
    try:
        ctx = request.app.state.ctx
        client = ctx.mlflow_client
        # TODO: is client necessary
        list_of_exps = get_experiments(client)
        return HistoryResponse(
            exps=list_of_exps
        )
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )
    
# TODO: add endpoint that returns runs inside the experiment

@router.get('/{mlflow_run_id}', response_model=ResultsResponse)
async def get_run_results(request: Request, mlflow_run_id: str):
    try:
        ctx = request.app.state.ctx
        client = ctx.mlflow_client
        results = get_run_results(client, mlflow_run_id)
        return ResultsResponse(results=results)
    except Exception as e:
        print(traceback.format_exc())
        raise  HTTPException(
            status_code=500,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )
    
# TODO: add endpoints later to separately load metrics, cfg, error analysis
# TODO: add endpoint for deleting run from mlflow 