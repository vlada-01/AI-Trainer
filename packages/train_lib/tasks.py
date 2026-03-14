from enum import Enum

class AvailableTasks(str, Enum):
    classification = 'classification'
    regression = 'regression'
    bbox = 'bbox'