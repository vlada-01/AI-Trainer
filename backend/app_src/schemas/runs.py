from pydantic import BaseModel
from typing import Optional, Literal, Any, List

from app_src.schemas.job_response import ErrorInfo, JobResponse

class RunCtxResponse(BaseModel):
    id: str
    status: Literal['draft', 'ready', 'running']
    required_steps: Optional[List[Literal['dataset, predictor, train_params']]] = None
    status_details: Optional[Any] = None
    jobs: Optional[List[JobResponse]] = None
    error: Optional[ErrorInfo] = None
    created_at: str
    updated_at: str

class NewRunCfg(BaseModel):
    name: None