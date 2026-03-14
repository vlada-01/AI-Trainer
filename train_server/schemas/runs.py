from pydantic import BaseModel
from typing import List, Dict, Literal, Union

from train_server.schemas.job_response import ErrorInfo, JobResponse

from train_server.services.runs.state_manager import AvailableRunTypes
from packages.train_lib.tasks import AvailableTasks

class RunCtxResponse(BaseModel):
    run_id: str
    run_type: str
    state: str
    # required_steps: Optional[List[Literal['dataset, predictor, train_params']]] = None
    jobs: List[JobResponse]
    # error: Optional[ErrorInfo] = None
    created_at: str
    updated_at: str

class SpecsCfg(BaseModel):
    task: Union[
        Literal[AvailableTasks.classification],
        Literal[AvailableTasks.regression],
        Literal[AvailableTasks.bbox]
    ]

class NewRunCfg(BaseModel):
    specs: Dict[str, SpecsCfg]
    run_type: AvailableRunTypes