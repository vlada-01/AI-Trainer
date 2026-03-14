from pydantic import BaseModel, Field
from typing import Literal, Union, List, Dict, Any

from packages.train_lib.prepare_train.metrics.metric import AvailableMetrics

from train_server.schemas.data import TransformStep
from train_server.schemas.models import Layers


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

# thresholds in [0..1]


# class FtDatasetCfg(BaseModel):
#     new_train_transform: Optional[List[TransformStep]] = None

# # TODO: update this according to DAG
# class FTLayersDetails(BaseModel):
#     type: Union[Literal['backbone', 'new']]
#     freeze: bool = False
#     original_id: Optional[int] = None
    
# class FtLayersCfg(BaseModel):
#     use_torch_layers: Optional[bool] = False
#     layers: Layers
#     ft_layers_details: List[FTLayersDetails]

# class FtTrainCfg(BaseModel):
#     epochs: int
#     num_of_iters: int = 1
#     optimizer: OptimizerConfig
#     lr_decay: Optional[LrDecay] = None
#     loss_fn: LossFnConfig