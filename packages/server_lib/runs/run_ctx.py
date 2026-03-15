import os
import asyncio
from dataclasses import dataclass
from uuid import uuid4
from datetime import datetime, timezone
from enum import Enum

import train_server.schemas.job_request as requests

from torch.utils.data import DataLoader
from packages.train_lib.prepare_model.models.model.model import Model 
from packages.train_lib.prepare_train.engines.engine_manager import EngineManager
from packages.train_lib.meta import Meta

from packages.server_lib.runs.state_mgrs.state_mgr import StateManager
from packages.server_lib.runs.state_mgrs.builder import create_state_mgr
from packages.server_lib.runs.job_mgr import JobManager, create_job_mgr

from packages.logger.logger import get_logger

log = get_logger(__name__)

runs_inactivity = int(os.getenv("RUNS_INACTIVITY", 1800))
cleanup_jobs_interval = int(os.getenv("CLEANUP_JOBS_INTERVAL", "60"))

@dataclass
class RunTime:
    train: DataLoader = None
    val: DataLoader = None
    test: DataLoader = None
    model: Model = None
    engine: EngineManager = None
    mlflow_run_id: str = None

@dataclass
class Configs:
    dl_cfg: requests.PrepareDatasetJobRequest = None
    model_cfg: requests.PrepareModelJobRequest = None
    train_cfg: requests.PrepareTrainJobRequest = None

class AvailableRunTypes(Enum):
    base = 'base'
    fine_tune = 'fine_tune'
    post_process = 'post_process'
    final_evaluation = 'final_evaluation'

def create_run_ctx(run_cfg):
    run_type = run_cfg.run_type
    specs = run_cfg.specs
    log.info(f'Initializing new RunContext for type: {run_type}')
    run_ctx = RunContext(run_type, specs)
    return run_ctx

class RunContext:
    def __init__(self, run_type, specs):
        self.run_type: AvailableRunTypes = run_type
        self.run_id = uuid4().hex
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.updated_at = now

        self.state_mgr: StateManager = create_state_mgr(run_type)

        self.job_mgr: JobManager = create_job_mgr()
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
            kwargs = {
                'run_id': self.run_id,
                'run_type': self.run_type,
                'state': self.state_mgr.curr_state,
                # 'required_steps': self.required_steps,
                'jobs': [j.to_dict() for j in self.job_mgr.get_jobs()],
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat(),
            }
            return kwargs

    async def is_valid_to_add(self, state_code):
        async with self.run_ctx_lock:
            return self.state_mgr.is_valid_state(state_code)
        
    async def move_state(self, job_id):
        async with self.run_ctx_lock:
            job = self.job_mgr.get_job(job_id)
            self.state_mgr.move_state(job.job_type)

    async def add_job(self, job):
        async with self.run_ctx_lock:
            self.job_mgr.add_job(job)

    async def update_job(self, job_id, **kwargs):
        async with self.run_ctx_lock:
            self.job_mgr.update_job(job_id, **kwargs)
    
    async def get_job(self, job_id):
        async with self.run_ctx_lock:
            return self.job_mgr.get_job(job_id)

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
            finished = self.state_mgr.is_finished()
            is_running = self.state_mgr.is_running()
            now = datetime.now(timezone.utc)
            is_inactive = (now - self.updated_at) > runs_inactivity
            
            return finished or (not is_running and is_inactive)

    async def cleanup_task_loop(self):
        while True:
            await asyncio.sleep(self.cleanup_jobs_interval)
            async with self.run_ctx_lock:
                self.job_mgr.remove_jobs()

    async def cancel_cleanup_task(self):
        t = getattr(self, "cleanup_task", None)
        if not t or t.done():
            return
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass