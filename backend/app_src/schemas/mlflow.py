from pydantic import BaseModel
from typing import List, Any, Dict

class Experiment(BaseModel):
    name: str
    url: str

class HistoryResponse(BaseModel):
    exps: List[Experiment]

class ResultsResponse(BaseModel):
    results: Dict[str, Any]