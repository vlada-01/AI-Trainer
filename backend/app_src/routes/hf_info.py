import traceback
from fastapi import APIRouter, Request, HTTPException

from app_src.services.hf_info import get_dataset_info
from app_src.schemas.hf_info import DatasetInfoRequest, DatasetInfoResponse
from app_src.schemas.job_response import ErrorInfo

from common.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/data-info", tags=["hf_info"])

@router.post('/get-dataset-info', response_model=DatasetInfoResponse)
def get_ds_info(request: Request, data: DatasetInfoRequest):
    try:
        log.info('Requesting dataset info')
        ds_info_dict = get_dataset_info(data)
        log.info('Dataset Info succesfully found')
        return DatasetInfoResponse(
            status='success',
            status_details=ds_info_dict
        )
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e),
            )
        )