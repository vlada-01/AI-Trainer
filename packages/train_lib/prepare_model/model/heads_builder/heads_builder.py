from packages.train_lib.tasks import AvailableTasks

from .heads import Classification
from .heads import Regression
from .heads import BBox 

HEADS_REGISTRY_MAP = {
    AvailableTasks.classification: Classification,
    AvailableTasks.regression: Regression,
    AvailableTasks.bbox: BBox,
}

def build_heads(meta):
    heads = {}
    for k, spec in meta.get_specs().items():
        task = spec.task
        if task not in HEADS_REGISTRY_MAP:
            raise ValueError(f'Head does not exist for the task: {task.value}')
        heads[k] = HEADS_REGISTRY_MAP[task](task)
    return heads
