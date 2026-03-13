from pydantic import BaseModel
from typing import Literal, Dict, Optional, Any

class DatasetInfoRequest(BaseModel):
    id: str
    name: Optional[str] = None

class DatasetInfoResponse(BaseModel):
    status: Literal['success']
    status_details: Dict[str, Any]