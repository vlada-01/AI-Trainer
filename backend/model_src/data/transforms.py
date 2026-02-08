import torch.nn.functional as F
import torchvision.transforms as T
import torch
from enum import Enum
from pprint import pformat

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

# TODO: consider adding raw transformation that takes first and last 128 tokens for example. Good for imdb
def build_text_to_tensor(params):
    vocab = params['vocab']
    max_len = params['max_len']
    unk_id = vocab['<unk>']
    
    def tokenize_data(tokens):
        vocab_tokens = ['<sos>']
        vocab_tokens.extend(tokens[:max_len - 1])
        ids = [vocab.get(tok, unk_id) for tok in vocab_tokens]

        tenzorized = torch.tensor(ids)
        ones = torch.ones(len(tenzorized))
        zeros = torch.tensor([0] * (max_len - len(tenzorized)))
        mask = torch.cat([ones, zeros])
        padding = torch.tensor([vocab['<pad>']] * (max_len - len(tenzorized)))
        return {
            'input_ids': torch.cat([tenzorized, padding]),
            'attention_mask': mask
        }

    return T.Lambda(lambda x: tokenize_data(x))

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
    text_to_tensor = 'text_to_tensor'
    to_tensor = 'to_tensor'

REGISTRY_TRANSFORMS_MAP = {
    AvailableTransforms.normalize_l1_l2: build_normalize_l1_l2,
    AvailableTransforms.normalize_mean_std: build_normalize_mean_std,
    AvailableTransforms.normalize_min_max: build_normalize_min_max,
    AvailableTransforms.random_horizontal_flip: build_random_horizontal_flip,
    AvailableTransforms.random_rotation: build_random_rotation,
    AvailableTransforms.random_vertical_flip: build_random_vertical_flip,
    AvailableTransforms.img_to_tensor: build_img_to_tensor,
    AvailableTransforms.to_tensor: build_to_tensor,
    AvailableTransforms.text_to_tensor: build_text_to_tensor
}

def assemble_transforms(cfg_dataset_transforms, meta):
    train_t, train_tt = None, None
    val_t, val_tt = None, None
    test_t, test_tt = None, None

    cfg = cfg_dataset_transforms
    if cfg is not None:
        t_cfg = cfg.train.transform
        tt_cfg = cfg.train.target_transform
        log.debug('Composing train transforms')
        train_t = compose_transforms(t_cfg, meta)
        log.debug('Composing train target transforms')
        train_tt = compose_transforms(tt_cfg, meta)

        t_cfg = cfg.valid.transform 
        tt_cfg = cfg.valid.target_transform
        log.debug('Composing val transforms') 
        val_t = compose_transforms(t_cfg, meta)
        log.debug('Composing val target transforms')
        val_tt = compose_transforms(tt_cfg, meta)

        t_cfg = cfg.test.transform if cfg.test is not None else None
        tt_cfg = cfg.test.target_transform if cfg.test is not None else None
        log.debug('Composing test transforms') 
        test_t = compose_transforms(t_cfg, meta)
        log.debug('Composing test target transforms') 
        test_tt = compose_transforms(tt_cfg, meta)
    return train_t, train_tt, val_t, val_tt, test_t, test_tt

def compose_transforms(cfg, meta):
    if cfg is None:
        return None
    log.debug('Assembling transforms for the cfg:\n%s', pformat(cfg))
    normalized = normalize_params(cfg)
    log.debug('Normalized parameters:\n%s', pformat(normalized))
    resolved = resolve_params(normalized, meta)
    log.debug('Resolved parameters:\n%s', pformat({name: type(v).__name__ for name, v in resolved}))
    tfs = generate_transforms(resolved)
    log.debug('Assembled transformations.Compose:\n%s', pformat(tfs.transforms))
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

