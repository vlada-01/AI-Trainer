from pydantic import BaseModel, Field
from typing import Literal, Union, Tuple, Optional, List, Any

from model_src.models.model_builder import AvailableForwardTypes
from model_src.models.layers.layer_factory import AvailableLayers

#----------------------------------
class BatchNorm2dLayer(BaseModel):
    type: Literal[AvailableLayers.batchnorm2d]
    num_features: int

class Conv2DLayer(BaseModel):
    type: Literal[AvailableLayers.conv2d]
    in_channels: int
    out_channels: int
    kernel_size: Union[int, Tuple[int, int]]
    padding: int = 0
    stride: int = 1

class MaxPool2DLayer(BaseModel):
    type: Literal[AvailableLayers.maxpool]
    kernel_size: Union[int, Tuple[int, int]]
    stride: Optional[int] = None

class LinearLayer(BaseModel):
    type: Literal[AvailableLayers.linear]
    in_features: int
    out_features: int
    bias: bool = True

class ReLULayer(BaseModel):
    type: Literal[AvailableLayers.relu]

class DropoutLayer(BaseModel):
    type: Literal[AvailableLayers.dropout]
    p: float = Field(..., le=1.0, ge=0.0)

class FlattenLayer(BaseModel):
    type: Literal[AvailableLayers.flatten]

class PositionalEmbedding(BaseModel):
    type: Literal[AvailableLayers.positional_embedding]
    vocab_size: int
    emb_dim: int
    max_len: int
    padding_idx: int
    dropout: float

class Pooling(BaseModel):
    type: Literal[AvailableLayers.pooling]

class TransformerEncoder(BaseModel):
    type: Literal[AvailableLayers.transformer_encoder]
    num_layers: int
    emb_dim: int
    h: int
    ffn_size: int
    dropout: float

Layers = List[Union[
        BatchNorm2dLayer,
        Conv2DLayer,
        LinearLayer,
        ReLULayer,
        MaxPool2DLayer,
        DropoutLayer,
        FlattenLayer,
        PositionalEmbedding,
        Pooling,
        TransformerEncoder
    ]]

class ModelJobRequest(BaseModel):
    forward_type: Union[
        Literal[AvailableForwardTypes.default],
        Literal[AvailableForwardTypes.dict]
    ] 
    use_torch_layers: Optional[bool] = False
    layers: Layers

class ErrorInfo(BaseModel):
    error_type: str
    error_message: str
    traceback: Any

class ModelJobResponse(BaseModel):
    id: str
    status: Literal['pending', 'in_progress', 'success', 'failed']
    status_details: str = ''
    error: Optional[ErrorInfo] = None
    created_at: str
    expires_at: str

#----------------------------------

class Experiment(BaseModel):
    name: str
    url: str

class HistoryResponse(BaseModel):
    exps: List[Experiment]

#----------------------------------
