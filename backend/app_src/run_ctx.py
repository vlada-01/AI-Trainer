import asyncio
from torch.utils.data import DataLoader
from uuid import uuid4
from datetime import datetime, timezone, timedelta
from typing import Dict, Union, Optional

from app_src.schemas.runs import RunCtxResponse

import app_src.schemas.job_request as requests

from model_src.data.metas.meta import MetaData
from model_src.models.model_builder import Predictor

class RunContext:
    def __init__(self):
        self.run_id: str = uuid4().hex
        self.status: str = 'draft' 
        self.required_steps: list[str] = ['prepare_dataset, prepare_model, prepare_train']
        now = datetime.now(timezone.utc).isoformat()
        self.created_at: str = now
        self.updated_at: str = now
        self.jobs: Dict[str, Union[
            requests.DatasetJobRequest,
            requests.ModelJobRequest,
            requests.TrainJobRequest
            ]
        ]
        self.jobs_lock: asyncio.Lock
        
        # TODO: add later
        # ?CLEAN_UP_INTERVAL = 60
        # self.cleanup_task: asyncio.Task = asyncio.create_task(utils.cleanup_jobs_loop(ctx, CLEAN_UP_INTERVAL))
        
        self.predictor: Optional[Predictor] = None
        self.train: Optional[DataLoader] = None
        self.val: Optional[DataLoader] = None
        self.test: Optional[DataLoader] = None
        self.meta: Optional[MetaData] = None

        # cache
        self.cached_model_cfg: Optional[requests.ModelJobRequest] = None
        self.cached_pp_cfg: Optional[requests.PostProcessingJobRequest] = None
        self.cached_dl_cfg: Optional[requests.DatasetJobRequest] = None
        self.cached_train_cfg: Optional[requests.TrainJobRequest] = None
        self.cached_run_id: Optional[str] = None

    def get_info(self):
        kwargs = {
            'run_id': self.run_id,
            'status': self.status,
            'required_steps': self.required_steps,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }
        return RunCtxResponse(**kwargs)
    
    def update(self, result):
        for k, v in result.items():
                if hasattr(self, k):
                    setattr(self, k, v)
                else:
                    raise ValueError(f'Field {k} does not exist in the AppContext')
                
    def get_train_params(self):
         return (
              self.predictor,
              self.train,
              self.val,
              self.meta,
              self.cached_dl_cfg,
              self.cached_model_cfg
         )