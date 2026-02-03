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

class TrainCfg(BaseModel):
    device: str = 'cpu'

    epochs: int
    num_of_iters: int

    optimizer: OptimizerConfig
    lr_decay: Optional[LrDecay] = None

    loss_fn: LossFnConfig
    
    metrics: List[Union[
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

class TrainJobRequest(BaseModel):
    exp_name: str
    run_name: str
    model_name: str
    log_train_metrics: Optional[bool] = False
    train_cfg: TrainCfg

class ErrorInfo(BaseModel):
    error_type: str
    error_message: str
    traceback: Any

class JobResponse(BaseModel):
    id: str
    status: Union[Literal['sucess', 'pending', 'in_progress', 'failed']]
    status_details: Optional[Any] = None
    error: Optional[ErrorInfo] = None
    created_at: str
    expires_at: str

class InspectRunJobRequest(BaseModel):
    run_id: str

# thresholds in [0..1]
class GlobalThreshold(BaseModel):
    type: Literal[AvailablePostProcessors.global_threshold]
    accuracy: float
    threshold: Optional[float] = None

class Calibration(BaseModel):
    type: Literal[AvailablePostProcessors.calibration]
    T: Optional[float] = None

class PostProcessingJobRequest(BaseModel):
    new_run_name: str
    post_processors: List[Union[
        Calibration,
        GlobalThreshold
    ]]

class FtDatasetCfg(BaseModel):
    new_train_transform: Optional[List[TransformStep]] = None

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

class FineTuneJobRequest(BaseModel):
    new_run_name: str
    new_ds_cfg: FtDatasetCfg
    new_layers_cfg: FtLayersCfg
    new_train_cfg: FtTrainCfg