from fastapi import APIRouter, Request, HTTPException
import asyncio

import app_src.utils.data  as utils
from app_src.schemas.data import DatasetJobRequest, DatasetJobResponse
from app_src.schemas.data import DatasetInfoRequest, DatasetInfoResponse

from app_src.services.data import create_job, get_job, run_prepare_dataset

from common.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/data", tags=["dataloaders"])

@router.post('/get_dataset_info', response_model=DatasetInfoResponse)
def get_ds_info(request: Request, data: DatasetInfoRequest):
    try:
        log.debug(f'Requesting get_ds_info for {data.model_dump()}')
        ds_info_dict = utils.get_dataset_info(data)
        log.info('Dataset Info succesfully found')
        return DatasetInfoResponse(
            status='success',
            status_details=ds_info_dict
        )
    except Exception as e:
        log.info(f'{type(e).__name__}, {str(e)}')
        return DatasetInfoResponse(
            status='failed',
            status_details='Failed to load Dataset info',
            error={
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
        )
# TODO: Need to test All transformation types
@router.post('/prepare_dataset', response_model=DatasetJobResponse)
async def prepare_dataset(request: Request, data: DatasetJobRequest):
    ctx = request.app.state.ctx
    job = await create_job(ctx)

    asyncio.create_task(run_prepare_dataset(ctx, job.id, data))
    return job

@router.get("/dataset_status/{job_id}")
async def dataset_status(request: Request, job_id: str):
    ctx = request.app.state.ctx

    job = await get_job(ctx, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"error": "job_not_found"})

    return job
