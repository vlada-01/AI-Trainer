from pydantic import BaseModel
from typing import Optional, Literal, Any, List

from app_src.schemas.job_response import ErrorInfo

class RunCtxResponse(BaseModel):
    id: str
    status: Literal['draft']
    required_steps: List[Literal['dataset, predictor, train_params']]
    status_details: Optional[Any] = None
    error: Optional[ErrorInfo] = None
    created_at: str
    updated_at: str

class NewRunCfg(BaseModel):
    name: None


