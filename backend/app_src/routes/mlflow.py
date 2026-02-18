import traceback
from fastapi import APIRouter, Request, HTTPException

from app_src.schemas.mlflow import HistoryResponse, ResultsResponse, ExperimentRunsResponse
from app_src.schemas.job_response import ErrorInfo

from app_src.services.mlflow import get_experiments, get_run_results, get_runs, delete_run

from common.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/mlflow", tags=["mlflow"])

@router.get('/history', response_model=HistoryResponse)
def get_history(request: Request):
    try:
        ctx = request.app.state.ctx
        client = ctx.mlflow_client
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

@router.get('/get-exp-runs/{exp_name}', response_model=ExperimentRunsResponse)
def get_exp_runs(request: Request, exp_name: str):
    try:
        ctx = request.app.state.ctx
        client = ctx.mlflow_client
        runs_list = get_runs(client, exp_name)
        return ExperimentRunsResponse(
            runs=runs_list
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

@router.get('/{mlflow_run_id}', response_model=ResultsResponse)
def get_run_results(request: Request, mlflow_run_id: str):
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

@router.delete('/{mlflow_run_id}', status_code=204)
def delete_mlflow_run(request: Request, mlflow_run_id: str):
    try:
        ctx = request.app.state.ctx
        client = ctx.mlflow_client
        delete_run(client, mlflow_run_id)
        return None
    except Exception as e:
        print(traceback.format_exc())
        raise  HTTPException(
            status_code=500,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e)
            )
        )