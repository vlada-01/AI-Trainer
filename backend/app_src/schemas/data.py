from pydantic import BaseModel, Field, model_validator
from typing import Literal, Union, Optional, Dict, Any, List

from model_src.data.metas.meta import  MetaTypes
from model_src.data.dataset_builders.builder import AvailableProviders
from model_src.data.transforms import AvailableTransforms
from model_src.data.mapper import AvailableMappers

class L1L2Norm(BaseModel):
    p: int = Field(..., ge=1, le=2)

class Probability(BaseModel):
    p: float = Field(..., ge=0.0, le=1.0)

class Rotation(BaseModel):
    alpha: int

class NormalizeL1L2(BaseModel):
    name: Literal[AvailableTransforms.normalize_l1_l2]
    value: L1L2Norm

class NormalizeMeanStd(BaseModel):
    name: Literal[AvailableTransforms.normalize_mean_std]
    value: Optional[bool] = True

class NormalizeMinMax(BaseModel):
    name: Literal[AvailableTransforms.normalize_min_max]
    value: Optional[bool] = True

class RandomHorizontalFlip(BaseModel):
    name: Literal[AvailableTransforms.random_horizontal_flip]
    value: Probability

class RandomVerticalFlip(BaseModel):
    name: Literal[AvailableTransforms.random_vertical_flip]
    value: Probability

class RandomRotation(BaseModel):
    name: Literal[AvailableTransforms.random_rotation]
    value: Rotation

class ImgToTensor(BaseModel):
    name: Literal[AvailableTransforms.img_to_tensor]

class TextToTensor(BaseModel):
    name: Literal[AvailableTransforms.text_to_tensor]

class ToTensor(BaseModel):
    name: Literal[AvailableTransforms.to_tensor]

TransformStep = Union[
    NormalizeL1L2,
    NormalizeMeanStd,
    NormalizeMinMax,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    ImgToTensor,
    TextToTensor,
    ToTensor
]

TargetTargetTransformStep = Union[
    ToTensor
]

class TransformConfig(BaseModel):
    transform: List[TransformStep]
    target_transform: List[TargetTargetTransformStep]

class DataTransforms(BaseModel):
    train: TransformConfig
    valid: TransformConfig
    test: TransformConfig

class SklearnConfig(BaseModel):
    dataset_provider: Literal[AvailableProviders.sklearn]
    dataset_fn: str
    task: Union[Literal['classification', 'regression']]
    stratify: Optional[bool] = True
    test_size: float = Field(..., le=1.0, ge=0.0)
    val_size: float = Field(..., le=1.0, ge=0.0)
    
    @model_validator(mode="after")
    def check_split_sizes(self):
        if self.test_size is not None and self.test_size > 0.5:
            raise ValueError(f'{self.test_size} > 0.5 is not okay')
        elif self.test_size is not None:
            if self.val_size > (1.0 - self.test_size) / 2:
                raise ValueError(f'{self.val_size} > {(1.0 - self.test_size) / 2} can not be greater than half of the remaining portion')
        return self

class SimpleMapper(BaseModel):
    name: Literal[AvailableMappers.simple]
    x_mapping: str
    y_mapping: str

class HuggingFaceConfig(BaseModel):
    dataset_provider: Literal[AvailableProviders.hf]
    id: str
    name: Optional[str] = None
    task: Union[Literal['classification', 'regression']]
    meta_type: Union[
        Literal[MetaTypes.tabular], #TODO: need to add this support
        Literal[MetaTypes.image],
        Literal[MetaTypes.textual]
        ]
    train_split: str
    val_split: str
    test_split: str
    load_ds_args: Dict[str, Any] #TODO: can be improved
    mapper: Union[
        SimpleMapper
    ]

