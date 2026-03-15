import os
import asyncio
from dataclasses import dataclass
from torch.utils.data import DataLoader
from uuid import uuid4
from datetime import datetime, timezone
from typing import Dict, Optional

from train_server.schemas.job_response import JobResponse
import train_server.schemas.job_request as requests

from packages.train_lib.prepare_model.models.model.model import Model 
from packages.train_lib.prepare_train.engines.engine_manager import EngineManager

from packages.train_lib.meta import Meta

from train_server.services.runs.state_manager import AvailableRunTypes, get_state_mappings, StateCode

runs_inactivity = int(os.getenv("RUNS_INACTIVITY", 1800))
cleanup_jobs_interval = int(os.getenv("CLEANUP_JOBS_INTERVAL", "60"))

@dataclass
class RunTime:
    train: Optional[DataLoader] = None
    val: Optional[DataLoader] = None
    test: Optional[DataLoader] = None
    model: Optional[Model] = None
    engine: Optional[EngineManager] = None
    mlflow_run_id: Optional[str] = None

@dataclass
class Configs:
    dl_cfg: Optional[requests.PrepareDatasetJobRequest] = None
    model_cfg: Optional[requests.PrepareModelJobRequest] = None
    train_cfg: Optional[requests.PrepareTrainJobRequest] = None

class RunContext:
    def __init__(self, run_type, specs):
        self.run_type: AvailableRunTypes = run_type
        self.state_mapping = get_state_mappings(run_type)
        self.run_id: str = uuid4().hex
        self.state: StateCode = StateCode.draft
        now = datetime.now(timezone.utc)
        self.created_at: str = now
        self.updated_at: str = now

        self.jobs: Dict[str, JobResponse] = {}
        self.run_ctx_lock: asyncio.Lock = asyncio.Lock()
        
        self.cleanup_jobs_interval = cleanup_jobs_interval
        self.cleanup_task: asyncio.Task = asyncio.create_task(self.cleanup_task_loop())
        
        # stored data 
        self.meta = Meta(specs)
        self.runtime = RunTime()
        self.cfgs = Configs()

    # TODO: add meta data
    async def get_info(self):
        async with self.run_ctx_lock:
            jobs = [v for v in self.jobs.values()]
            kwargs = {
                'run_id': self.run_id,
                'run_type': self.run_type,
                'state': self.state.name,
                # 'required_steps': self.required_steps,
                'jobs': jobs,
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat(),
            }
            return kwargs

    async def is_valid_to_add(self, status_code):
        async with self.run_ctx_lock:
            return status_code in self.state_mapping[self.state] 

    async def update(self, result):
        async with self.run_ctx_lock:
            # FIXME: update this check
            if not all(hasattr(self, k) for k in result.keys()):
                raise ValueError(f'Not all fields exist in the RunContext')
            for k, v_dict in result.items():
                    obj = getattr(self, k)
                    for attr, v in v_dict.items():
                        setattr(obj, attr, v)
                        
            self.updated_at = datetime.now(timezone.utc)

    async def move_state(self, job_id):
        async with self.run_ctx_lock:
            job = self.jobs[job_id]
            self.state = job.job_type

    async def get_prepare_model_params(self):
        async with self.run_ctx_lock:
            return(
                self.meta,
            )

    async def get_prepare_engine_params(self):
        async with self.run_ctx_lock:
            return (
                self.runtime.model,
                self.runtime.train,
                self.runtime.val,
                self.runtime.test,
                self.meta)

    async def get_train_params(self):
         async with self.run_ctx_lock:
            return (
                self.runtime.engine,
                self.meta,
                self.cfgs
            )
         
    async def get_prepare_complete_train_params(self):
         async with self.run_ctx_lock:
             return (
                 self.meta,
             )
         
    async def get_prepare_default_from_run_params(self):
        async with self.run_ctx_lock:
             return (
                 self.meta,
             )
         
    async def get_post_process_params(self):
        async with self.run_ctx_lock:
            return (
                self.runtime.engine,
                self.meta,
                self.cfgs.model_cfg,
            )
         
    async def get_final_eval_params(self):
        async with self.run_ctx_lock:
            return (
                self.runtime.engine,
                self.runtime.mlflow_run_id,
                self.meta,
                self.cfgs
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
                    if (job.expires_at and datetime.fromisoformat(job.expires_at) <= now)
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