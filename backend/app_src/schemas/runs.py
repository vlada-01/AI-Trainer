from pydantic import BaseModel
from typing import Optional, Literal, Any, List

from app_src.schemas.job_response import ErrorInfo, JobResponse

from app_src.services.runs.state_manager import AvailableRunTypes

class RunCtxResponse(BaseModel):
    run_id: str
    run_type: str
    state: str
    # required_steps: Optional[List[Literal['dataset, predictor, train_params']]] = None
    jobs: List[JobResponse]
    # error: Optional[ErrorInfo] = None
    created_at: str
    updated_at: str

class NewRunCfg(BaseModel):
    run_type: AvailableRunTypes