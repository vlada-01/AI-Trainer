from pydantic import BaseModel
from typing import Optional, Literal, Any, List

from app_src.schemas.job_response import ErrorInfo, JobResponse

from app_src.services.runs.state_manager import AvailableRunTypes, StateCode

class RunCtxResponse(BaseModel):
    id: str
    state: StateCode
    # required_steps: Optional[List[Literal['dataset, predictor, train_params']]] = None
    status_details: Optional[Any] = None
    jobs: Optional[List[JobResponse]] = None
    error: Optional[ErrorInfo] = None
    created_at: str
    updated_at: str

class NewRunCfg(BaseModel):
    run_type: AvailableRunTypes