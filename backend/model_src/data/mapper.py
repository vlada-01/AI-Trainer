from enum import Enum

class AvailableMappers(str, Enum):
    simple = 'simple'

def simple(mapper_cfg):
    x_key = mapper_cfg.x_mapping
    y_key = mapper_cfg.y_mapping
    return lambda sample: (sample[x_key], sample[y_key])

MAPPER_MAP = {
    AvailableMappers.simple: simple
}

def prepare_mapper(mapper_cfg):
    name = mapper_cfg.name
    if name not in MAPPER_MAP:
        raise AttributeError(f'There is no mapping name {name} in registry0')
    return MAPPER_MAP[name](mapper_cfg)

