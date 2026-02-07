from enum import Enum
from torch import nn
import torch

from model_src.models.layers.layer_factory import build_layers, build_layer
from model_src.models.post_processor import build_post_processor

from common.logger import get_logger

log = get_logger(__name__)

# TODO: later will be needed specific forward types due to eval
class AvailableForwardTypes(str, Enum):
    default = 'default_forward'
    dict = 'dictionary_forward'

def build_predictor(model_cfg, pp_cfg=None):
    model = build_model(model_cfg)
    pp = build_post_processor(pp_cfg) if pp_cfg is not None else None
    predictor = Predictor(model, pp)
    return predictor

def build_model(cfg):
    forward_type = cfg.forward_type
    use_torch_layers = cfg.use_torch_layers
    layers_cfg = cfg.layers
    log.info(f'Initialzing model builder for the {cfg}')
    model = Net(forward_type, layers_cfg, use_torch_layers)
    log.info('Model is prepared')
    return model

def prepare_fine_tune_model(predictor, new_layers_cfg):
    use_torch_layers = new_layers_cfg.use_torch_layers
    layers = new_layers_cfg.layers
    ft_layers = new_layers_cfg.ft_layers_details

    model = predictor.get_model()

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
    predictor.set_model(model)
    return predictor

def update_model_cfg(old_cfg, new_layers_cfg):
    old_cfg.layers = new_layers_cfg.layers
    return old_cfg
            
# TODO: later will need more classes for model types that will have predict method, for example DAG support.
class Net(nn.Module):
    def __init__(self, forward_type, layers_cfg, use_torch_layers):
        super().__init__()
        self.forward_fn = self.select_forward_fn(forward_type)
        self.layers = nn.Sequential(*build_layers(layers_cfg, use_torch_layers))

    def forward(self, x):
        return self.forward_fn(x)
    
    def set_pp(self, pp):
        self.pp = pp

    def get_pp(self):
        return self.pp
    
    def select_forward_fn(self, forward_type):
        if hasattr(self, forward_type.value):
            if callable(getattr(self, forward_type.value)):
                fn = getattr(self, forward_type.value)
                return fn
            else:
                raise ValueError(f'Net only supports callable, but got {forward_type.value}')
        else:
            raise ValueError(f'Net does not support forward_type {forward_type.value}')
        
    def default_forward(self, x):
        return self.layers(x)
    
    # used only for textuals
    def dict_forward(self, x):
        X = x['input_ids']
        attn_mask = x['attention_mask']
        return self.layers(X, attn_mask)

# TODO: needs to be updated to have support for the regression
# TODO: check if pp needs torch no grad
class Predictor:
    def __init__(self, model, pp=None):
        self.model = model
        self.pp = pp

    def set_model(self, model):
        self.model = model

    def set_pp(self, pp):
        self.pp = pp

    def get_model(self):
        return self.model
    
    def get_pp(self):
        return self.pp

    def get_model_parameters(self):
        return self.model.parameters()

    def logits(self, x):
        return self.model(x)
    
    def preds(self, logits):
        if self.pp is not None:
            return self.pp.post_process(logits)
        # can directly return argmax(dim=1)
        probs = torch.softmax(logits, dim=1)
        return probs.argmax(dim=1)
    
    # always converts tensors to numpy arrays, because df is pandas series
    def preds_with_error_analysis(self, logits, df):
        if self.pp is not None:
            new_df, finals = self.pp.post_process(logits, df)
            return new_df, finals
        return df, self.preds(logits)
