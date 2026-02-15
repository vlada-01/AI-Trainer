from pydantic import BaseModel
from typing import List, Any

class Experiment(BaseModel):
    name: str
    url: str

class HistoryResponse(BaseModel):
    exps: List[Experiment]

class ResultsResponse(BaseModel):
    artifacts: Any