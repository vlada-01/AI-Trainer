import os
import asyncio
from torch.utils.data import DataLoader
from uuid import uuid4
from datetime import datetime, timezone
from typing import Dict, Optional, List

from app_src.schemas.job_response import JobResponse
import app_src.schemas.job_request as requests

from model_src.data.metas.meta import MetaData
from model_src.models.model_builder import Predictor
from model_src.prepare_train.prepare_train import TrainParams

from app_src.services.runs.state_manager import AvailableRunTypes, get_state_mappings, StateCode

runs_inactivity = int(os.getenv("RUNS_INACTIVITY", 1800))
cleanup_jobs_interval = int(os.getenv("CLEANUP_JOBS_INTERVAL", "60"))

class RunContext:
    def __init__(self, run_type):
        self.run_type: AvailableRunTypes = run_type
        self.state_mapping = get_state_mappings(run_type)
        self.run_id: str = uuid4().hex
        self.state: StateCode = StateCode.draft
        # self.required_steps: List[]
        # # TODO: update this
        # self.required_steps: List[str] = ['prepare_dataset, prepare_model, prepare_train']
        now = datetime.now(timezone.utc)
        self.created_at: str = now
        self.updated_at: str = now

        self.jobs: Dict[str, JobResponse]
        self.run_ctx_lock: asyncio.Lock
        
        self.cleanup_jobs_interval = cleanup_jobs_interval
        self.cleanup_task: asyncio.Task = asyncio.create_task(self.cleanup_task_loop())
        
        # runtime objects required for run
        self.train: Optional[DataLoader] = None
        self.val: Optional[DataLoader] = None
        self.test: Optional[DataLoader] = None
        self.meta: Optional[MetaData] = None
        self.predictor: Optional[Predictor] = None
        self.train_params: Optional[TrainParams] = None

        # cached cfgs, stored as artifacts in the end of run
        self.cached_dl_cfg: Optional[requests.PrepareDatasetJobRequest] = None
        self.cached_model_cfg: Optional[requests.PrepareModelJobRequest] = None
        self.cached_train_cfg: Optional[requests.PrepareTrainJobRequest] = None
        self.cached_pp_cfg: Optional[requests.PreparePostProcessingJobRequest] = None
        self.cached_mlflow_run_id: Optional[str] = None

    async def get_info(self):
        async with self.run_ctx_lock:
            jobs = [v for v in self.jobs.values()]
            kwargs = {
                'run_id': self.run_id,
                'run_type': self.run_type,
                'state': self.state.name,
                # 'required_steps': self.required_steps,
                'jobs_status': jobs,
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat(),
            }
            return kwargs

    async def is_valid_to_add(self, status_code):
        async with self.run_ctx_lock:
            return status_code in self.state_mapping[self.state] 

    async def update(self, result):
        async with self.run_ctx_lock:
            for k, v in result.items():
                    if hasattr(self, k):
                        setattr(self, k, v)
                    else:
                        # TODO: update this crap, because some fields might be updated before exception
                        raise ValueError(f'Field {k} does not exist in the AppContext')
            self.updated_at = datetime.now(timezone.utc)
            # TODO: need to implement required_steps

    async def move_state(self, job_id):
        async with self.run_ctx_lock:
            job = self.jobs[job_id]
            self.state = self.state_mapping[job.job_type]

    async def get_prepare_train_params(self):
        async with self.run_ctx_lock:
            return (self.predictor, self.meta)

    async def get_train_params(self):
         async with self.run_ctx_lock:
            return (
                self.predictor,
                self.train,
                self.val,
                self.meta,
                self.train_params,
                self.cached_dl_cfg,
                self.cached_model_cfg,
                self.cached_train_cfg,
            )
         
    async def get_post_process_params(self):
        async with self.run_ctx_lock:
            return (
                self.predictor,
                self.val,
                self.train_params,
                self.meta,
                self.cached_dl_cfg,
                self.cached_model_cfg,
                self.cached_train_cfg,
                self.cached_mlflow_run_id
            )
         
    async def get_final_eval_params(self):
        async with self.run_ctx_lock:
            return (
                self.predictor,
                self.test,
                self.train_params,
                self.cached_mlflow_run_id
            )
    
    async def is_cleanable(self):
        async with self.run_ctx_lock:
            finished = self.state in (StateCode.done, StateCode.failed)
            still_running = self.state in (StateCode.training, StateCode.final_eval)

            now = datetime.now(timezone.utc)
            update_check = (now - self.updated_at) > runs_inactivity
            
            return finished or (not still_running and update_check)

    async def cleanup_task_loop(self):
        while True:
            await asyncio.sleep(self.cleanup_jobs_interval)
            now = datetime.now(timezone.utc)

            async with self.run_ctx_lock:
                expired_ids = [
                    job_id
                    for job_id, job in self.jobs.items()
                    if (job.expires_at and datetime.fromisoformat(job.expires_at) <= now) or job.status == 'success'
                ]
                for job_id in expired_ids:
                    del self.jobs[job_id]

    async def cancel_cleanup_task(self):
        t = getattr(self, "cleanup_task", None)
        if not t or t.done():
            return
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass