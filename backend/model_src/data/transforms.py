import torch.nn.functional as F
import torchvision.transforms as T
import torch
from enum import Enum

from common.logger import get_logger

log = get_logger(__name__)

def build_normalize_l1_l2(params):
    p = params['p']
    return T.Lambda(lambda x: F.normalize(x, p, dim=-1))

def build_normalize_mean_std(params):
    mean = torch.tensor(params['mean'])
    std = torch.tensor(params['std'])
    return T.Lambda(lambda x: (x - mean) / (std + 1e-8))

def build_normalize_min_max(params):
    min = torch.tensor(params['min'])
    max = torch.tensor(params['max'])
    return T.Lambda(lambda x: (x - min) / (max - min))

def build_img_to_tensor(params):
    return T.ToTensor()

def build_to_tensor(params):
    return T.Lambda(lambda x: torch.as_tensor(x))

def build_random_horizontal_flip(params):
    p = params['p']
    return T.RandomHorizontalFlip(p)

def build_random_vertical_flip(params):
    p = params['p']
    return T.RandomVerticalFlip(p)

def build_random_rotation(params):
    alpha = params['alpha']
    return T.RandomHorizontalFlip(alpha)



class AvailableTransforms(str, Enum):
    img_to_tensor = 'img_to_tensor'
    normalize_l1_l2 = 'normalize_l1_l2'
    normalize_mean_std = 'normalize_mean_std'
    normalize_min_max = 'normalize_min_max'
    random_horizontal_flip = 'random_horizontal_flip'
    random_rotation = 'random_rotation'
    random_vertical_flip = 'random_vertical_flip'
    to_tensor = 'to_tensor'

REGISTRY_TRANSFORMS_MAP = {
    AvailableTransforms.normalize_l1_l2: build_normalize_l1_l2,
    AvailableTransforms.normalize_mean_std: build_normalize_mean_std,
    AvailableTransforms.normalize_min_max: build_normalize_min_max,
    AvailableTransforms.random_horizontal_flip: build_random_horizontal_flip,
    AvailableTransforms.random_rotation: build_random_rotation,
    AvailableTransforms.random_vertical_flip: build_random_vertical_flip,
    AvailableTransforms.img_to_tensor: build_img_to_tensor,
    AvailableTransforms.to_tensor: build_to_tensor
    # TODO need to support some torchvision transformations
}

def compose_transforms(cfg, meta):
    if cfg is None:
        return None
    log.debug(f'Assembling transforms for the {cfg}')
    normalized = normalize_params(cfg)
    log.debug(f'Normalized parameters: {normalized}')
    resolved = resolve_params(normalized, meta)
    log.debug(f'Resolved parameters: {resolved}')
    tfs = generate_transforms(resolved)
    log.debug(f'Assembled transformations.Compose: {tfs.transforms}')
    return tfs

def normalize_params(cfg):
    normalized = []
    for step in cfg:
        if step.name in REGISTRY_TRANSFORMS_MAP:
            if isinstance(step.value, bool):
                if step.value:
                    normalized.append((step.name, {}))
            else:
                normalized.append((step.name, dict(step.value)))
        else:
            raise ValueError(f"Unknown transform: {step.name}")
    return normalized

def resolve_params(steps, meta):
    for i, step in enumerate(steps):
        if len(step[1]) == 0:
            steps[i] = meta.resolve(step[0])
    return steps

def generate_transforms(steps):
    tfs = []
    for name, params in steps:
        tf = REGISTRY_TRANSFORMS_MAP[name](params)
        tfs.append(tf)
    return T.Compose(tfs)

