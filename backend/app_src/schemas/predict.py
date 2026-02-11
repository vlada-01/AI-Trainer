from pydantic import BaseModel
from typing import Literal, Union, Optional, Any

class PredictJobRequest(BaseModel):
    run_id: str

class ErrorInfo(BaseModel):
    error_type: str
    error_message: str
    traceback: Any

class JobResponse(BaseModel):
    id: str
    status: Union[Literal['sucess', 'pending', 'in_progress', 'failed']]
    status_details: Optional[Any] = None
    error: Optional[ErrorInfo] = None
    created_at: str
    expires_at: str
