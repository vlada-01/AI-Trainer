from enum import Enum
from torch import nn

from model_src.models.layers.layer_factory import build_layers, build_layer

from common.logger import get_logger

log = get_logger(__name__)

# TODO: later will be needed specific model types due to eval
class AvailableModels(str, Enum):
    ffnn = 'ffnn'
    ffnn_custom = 'ffnn_custom'
    cnn = 'cnn'
    cnn_custom = 'cnn_custom'
    transformers = 'transfrormers'
    cnn_full = 'cnn_full'

# MODEL_BUILDER_MAP = {
#     AvailableModels.ffnn: FFNN
# }

def build_model(cfg):
    model_type = cfg.model_type
    use_torch_layers = cfg.use_torch_layers
    layers_cfg = cfg.layers
    log.info(f'Initialzing model builder for the {cfg}')
    # model = MODEL_BUILDER_MAP[model_type](layers_cfg, use_predefined_layers)
    model = Net(layers_cfg, use_torch_layers)
    log.info('Model is prepared')
    return model

def prepare_fine_tune_model(model, new_layers_cfg):
    use_torch_layers = new_layers_cfg.use_torch_layers
    layers = new_layers_cfg.layers
    ft_layers = new_layers_cfg.ft_layers_details

    old_layers = list(model.layers.children())
    new_layers = []
    for idx, ft_layer in enumerate(ft_layers):
        layer_type = ft_layer.type
        freeze = ft_layer.freeze
        original_id = ft_layer.original_id
        if layer_type == 'backbone':
            new_layers.append(old_layers[original_id])
            new_layers[-1].requires_grad = freeze
        elif layer_type == 'new':
            layer_cfg = layers[idx]
            cfg_dict = layers[idx].model_dump(exclude={'type', 'predefined'})
            new_layers.append(build_layer(layer_cfg.type, use_torch_layers, cfg_dict))
        else:
            log.error(f'{layer_type} does not exist for the Fine Tune layers field')
    log.info(f'new layers: {new_layers}')
    model.layers = nn.Sequential(*new_layers)
    log.info(f"Model: {model}")
    for name, p in model.named_parameters():
        print(name, p.shape, p.requires_grad)
    return model

def update_model_cfg(old_cfg, new_layers_cfg):
    old_cfg.layers = new_layers_cfg.layers
    return old_cfg

            
# TODO: later will need more classes for model types that will have predict method, 
class Net(nn.Module):
    def __init__(self, layers_cfg, use_torch_layers):
        super().__init__()
        self.layers = nn.Sequential(*build_layers(layers_cfg, use_torch_layers))

    def forward(self, x):
        return self.layers(x)