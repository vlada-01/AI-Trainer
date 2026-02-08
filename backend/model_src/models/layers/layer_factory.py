from torch import nn
from enum import Enum

import model_src.models.layers.layers as layers

from common.logger import get_logger

log = get_logger(__name__)

class AvailableLayers(str, Enum):
    batchnorm2d = 'BatchNorm2d'
    conv2d = 'Conv2d'
    dropout = 'Dropout'
    flatten = 'Flatten'
    linear = 'Linear'
    maxpool = 'MaxPool2d'
    pooling = 'Pooling'
    positional_embedding = 'PositionalEmbedding'
    relu = 'ReLU'
    transformer_encoder = 'TransformerEncoder'

def build_layers(layers_cfg, use_torch_layers):
    layers = []
    for layer in layers_cfg:
        type = layer.type
        log.debug(f'Constructing layer type: {type}')
        cfg_dict = layer.model_dump(exclude={'type', 'predefined'})
        layers.append(build_layer(type.value, use_torch_layers, cfg_dict))
    return layers

def build_layer(type, use_torch_layers, cfg_dict):
    layer_fn = None
    if use_torch_layers:
        layer_fn = getattr(nn, type, None)
    else:
        layer_fn = getattr(layers, type, None)
        if layer_fn is None:
             layer_fn = getattr(nn, type, None)
    
    if layer_fn is None:
            raise ValueError(f"Unknown layer type: {type}")
    
    return layer_fn(**cfg_dict)
     
    