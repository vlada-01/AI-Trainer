from fastapi import APIRouter, Request

import app_src.utils.data  as utils
from app_src.schemas.hf_info import DatasetInfoRequest, DatasetInfoResponse

from common.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/data-info", tags=["hf_info"])

@router.post('/get-dataset-info', response_model=DatasetInfoResponse)
def get_ds_info(request: Request, data: DatasetInfoRequest):
    try:
        log.debug(f'Initializing get dataset info for {data.model_dump()}')
        ds_info_dict = utils.get_dataset_info(data)
        log.info('Dataset Info succesfully found')
        return DatasetInfoResponse(
            status='success',
            status_details=ds_info_dict
        )
    except Exception as e:
        log.error(f'{type(e).__name__}, {str(e)}')
        return DatasetInfoResponse(
            status='failed',
            status_details='Failed to load Dataset info',
            error={
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
        )