from pydantic import BaseModel
from typing import Literal, Union, Optional, Any

from app_src.schemas.job_response import ErrorInfo

from model_src.data.dataset_builders.builder import AvailableProviders

class DatasetInfoRequest(BaseModel):
    dataset_provider: Literal[AvailableProviders.hf]
    id: str
    name: Optional[str] = None

class DatasetInfoResponse(BaseModel):
    status: Union[Literal['success', 'failed']]
    status_details: Optional[Any] = None
    error: Optional[ErrorInfo] = None
    hint: Optional[str] = None