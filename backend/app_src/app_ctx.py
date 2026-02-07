from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import asyncio
import mlflow
from torch.utils.data import DataLoader

from model_src.data.metas.meta import MetaData

from app_src.schemas.models import ModelJobRequest
from app_src.schemas.data import DatasetJobRequest
from app_src.schemas.train import TrainJobRequest, PostProcessingJobRequest

@dataclass
class AppContext:
    # required for initialization
    seed: int
    mlflow_client: mlflow.MlflowClient
    jobs: Dict[str, Union[
        DatasetJobRequest,
        ModelJobRequest,
        TrainJobRequest
        ]
    ]
    jobs_lock: asyncio.Lock

    cleanup_task: Optional[asyncio.Task] = None

    # run time objects
    predictor: Any = None
    train: Optional[DataLoader] = None
    val: Optional[DataLoader] = None
    test: Optional[DataLoader] = None
    meta: Optional[MetaData] = None

    # cache
    cached_model_cfg: Optional[ModelJobRequest] = None
    cached_pp_cfg: Optional[PostProcessingJobRequest] = None
    cached_dl_cfg: Optional[DatasetJobRequest] = None
    cached_train_cfg: Optional[TrainJobRequest] = None
    cached_run_id: Optional[str] = None
    