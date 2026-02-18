from pydantic import BaseModel
from typing import List, Any, Dict

class Experiment(BaseModel):
    name: str
    url: str

class HistoryResponse(BaseModel):
    exps: List[Experiment]

class Run(BaseModel):
    run_id: str
    url: str

class ExperimentRunsResponse(BaseModel):
    runs: List[Run]


class ResultsResponse(BaseModel):
    results: Dict[str, Any]