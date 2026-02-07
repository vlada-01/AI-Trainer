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

class TransformStep(BaseModel):
    name: Literal[
         AvailableTransforms.normalize_l1_l2,
        AvailableTransforms.normalize_mean_std,
        AvailableTransforms.normalize_min_max,
        AvailableTransforms.random_horizontal_flip,
        AvailableTransforms.random_rotation,
        AvailableTransforms.random_vertical_flip,
        AvailableTransforms.img_to_tensor,
        AvailableTransforms.text_to_tensor,
        AvailableTransforms.to_tensor
    ]
    value: Union[bool, L1L2Norm, Probability, Rotation]

class TargetTargetTransformStep(BaseModel):
    name: Literal[
        AvailableTransforms.to_tensor
    ]
    value: Union[bool]

class TransformConfig(BaseModel):
    transform: List[TransformStep]
    target_transform: List[TargetTargetTransformStep]

class DataTransforms(BaseModel):
    train: TransformConfig
    valid: TransformConfig
    test: Optional[TransformConfig] = None

class SklearnConfig(BaseModel):
    dataset_provider: Literal[AvailableProviders.sklearn]
    dataset_fn: str
    task: Union[Literal['classification', 'regression']]
    stratify: bool = True
    test_size: Optional[float] = None
    val_size: float = 0.2
    
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
        Literal[MetaTypes.tabular],
        Literal[MetaTypes.image],
        Literal[MetaTypes.textual]
        ]
    train_split: str
    val_split: str
    test_split: Optional[str] = None
    load_ds_args: Dict[str, Any]
    mapper: Union[
        SimpleMapper
    ]
    
class DatasetJobRequest(BaseModel):
    data_config: Union[SklearnConfig, HuggingFaceConfig]
    dataset_transforms: Optional[DataTransforms] = None
    batch_size: Optional[int] = 1
    shuffle: Optional[bool] = False
    

class ErrorInfo(BaseModel):
    error_type: str
    error_message: str
    traceback: Any

class DatasetJobResponse(BaseModel):
    id: str
    status: Literal['pending', 'in_progress', 'success', 'failed']
    status_details: Optional[Any] = None
    error: Optional[ErrorInfo] = None
    created_at: str
    expires_at: str

class DatasetInfoRequest(BaseModel):
    dataset_provider: Literal[AvailableProviders.hf]
    id: str
    name: Optional[str] = None

class DatasetInfoResponse(BaseModel):
    status: Union[Literal['success', 'failed']]
    status_details: Optional[Any] = None
    error: Optional[ErrorInfo] = None
    hint: Optional[str] = None