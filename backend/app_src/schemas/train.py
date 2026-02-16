from pydantic import BaseModel
from typing import Literal, Union, Optional, List, Dict, Any

from model_src.prepare_train.metrics import AvailableMetrics
from model_src.models.post_processor import AvailablePostProcessors

from app_src.schemas.data import TransformStep
from app_src.schemas.models import Layers


#----------------------------------
class LrDecay(BaseModel):
    type: str
    args: Dict[str, Any]

class OptimizerConfig(BaseModel):
    type: str
    args: Dict[str, Any]

class LossFnConfig(BaseModel):
    type: str
    args: Dict[str, Any]

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
class GlobalThreshold(BaseModel):
    type: Literal[AvailablePostProcessors.global_threshold]
    accuracy: float
    threshold: Optional[float] = None

class Calibration(BaseModel):
    type: Literal[AvailablePostProcessors.calibration]
    T: Optional[float] = None


class FtDatasetCfg(BaseModel):
    new_train_transform: Optional[List[TransformStep]] = None

# TODO: update this according to DAG
class FTLayersDetails(BaseModel):
    type: Union[Literal['backbone', 'new']]
    freeze: bool = False
    original_id: Optional[int] = None
    
class FtLayersCfg(BaseModel):
    use_torch_layers: Optional[bool] = False
    layers: Layers
    ft_layers_details: List[FTLayersDetails]

class FtTrainCfg(BaseModel):
    epochs: int
    num_of_iters: int = 1
    optimizer: OptimizerConfig
    lr_decay: Optional[LrDecay] = None
    loss_fn: LossFnConfig