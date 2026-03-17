from pydantic import BaseModel, Field
from typing import Literal, Union, List, Dict, Any

from ..prepare_train import AvailableMetrics

#----------------------------------
class LrDecay(BaseModel):
    type: str
    args: Dict[str, Any]

class OptimizerConfig(BaseModel):
    type: str
    args: Dict[str, Any]

class LossFn(BaseModel):
    type: str
    args: Dict[str, Any]

class LossFnCfg(BaseModel):
    weight: float = Field(..., le=1.0, ge=0.0)
    fn: LossFn
    

Metrics = List[Union[
        Literal[AvailableMetrics.accuracy],
        Literal[AvailableMetrics.precision],
        Literal[AvailableMetrics.recall],
        Literal[AvailableMetrics.f1_score]
        # Literal[AvailableMetrics.mse],
        # Literal[AvailableMetrics.mae],
        # Literal[AvailableMetrics.rmse],
        # Literal[AvailableMetrics.r2],
        # Literal[AvailableMetrics.bleu],
        # Literal[AvailableMetrics.perplexity],
    ]]