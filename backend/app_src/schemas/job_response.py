from pydantic import BaseModel
from typing import Literal, Optional, Any

class ErrorInfo(BaseModel):
    error_type: str
    error_message: str
    traceback: Optional[Any]

class JobResponse(BaseModel):
    id: str
    job_type: str
    status: Literal['pending', 'in_progress', 'success', 'failed']
    status_details: Optional[Any] = None
    error: Optional[ErrorInfo] = None
    created_at: str
    expires_at: str