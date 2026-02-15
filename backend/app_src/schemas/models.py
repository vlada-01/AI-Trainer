from pydantic import BaseModel, Field, model_validator
from typing import Literal, Union, Tuple, Optional, List

from model_src.models.model_types.dag import AvailableNodeTypes
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
    bias: Optional[bool] = True

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

# TODO: update usages in other places, fine-tune changed from List[Union] to Union
Layers = Union[
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
    ]

class InputCfg(BaseModel):
    type: Literal[AvailableNodeTypes.input]
    id: str
    in_keys: List[str]
    out_keys: List[str]

class LayerCfg(BaseModel):
    type: Literal[AvailableNodeTypes.layer]
    id: str
    layer_cfg: Layers
    in_keys: List[str]
    out_keys: List[str]

class ChainCfg(BaseModel):
    type: Literal[AvailableNodeTypes.chain]
    id: str
    layers_cfg: List[Layers]
    in_keys: List[str]
    out_keys: List[str]

class ComponentCfg(BaseModel):
    type: Literal[AvailableNodeTypes.component]
    id: str
    nodes: List[Union[InputCfg, LayerCfg, ChainCfg]]
    edges: List[Tuple[str, str]]
    in_keys: List[str]
    out_keys: List[str]
    
    @model_validator(mode='after')
    def validate_component(self):
        node_ids = set([node.id for node in self.nodes])

        # node ids must be unique
        if len(node_ids) != len(self.nodes):
            raise ValueError(f'Component configuration error: contains duplicates') 
            
        # all edge nodes must be in the nodes
        for u, v in self.edges:
            if u not in node_ids or v not in node_ids:
                raise ValueError(f'Component configuration error: Invalid edge ({u},{v})')

NodeCfg = Union[InputCfg, LayerCfg, ChainCfg, ComponentCfg]    

class DAGCfg(BaseModel):
    node_ids: List[str]
    edges: List[Tuple[str, str]]
    # TODO: extend latar so model can return more then one value
    out_key: str
