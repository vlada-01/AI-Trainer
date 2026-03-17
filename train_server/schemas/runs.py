from pydantic import BaseModel
from typing import List, Dict, Literal, Union, Optional

from .job_response import JobResponse

from packages.server_lib.runs import AvailableRunTypes
from packages.train_lib.tasks import AvailableTasks

class RunCtxResponse(BaseModel):
    run_id: str
    run_type: str
    state: str
    # required_steps: Optional[List[Literal['dataset, predictor, train_params']]] = None
    jobs: List[JobResponse]
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
    run_type: Optional[AvailableRunTypes] = None