from pydantic import BaseModel
from typing import Literal, Optional, Any

from packages.server_lib.runs import StateCode, JobStatus

class ErrorInfo(BaseModel):
    error_type: str
    error_message: str

class JobResponse(BaseModel):
    id: str
    job_type: StateCode
    status: Literal[
        JobStatus.pending,
        JobStatus.in_progress,
        JobStatus.success,
        JobStatus.failed
    ]
    status_details: Optional[Any] = None
    created_at: str
    expires_at: str