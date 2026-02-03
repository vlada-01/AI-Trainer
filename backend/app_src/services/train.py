# from __future__ import annotations

import asyncio
import traceback
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from app_src.app_ctx import AppContext
from app_src.schemas.train import TrainJobRequest, InspectRunJobRequest, PostProcessingJobRequest, FineTuneJobRequest, JobResponse, ErrorInfo

from app_src.utils.train import train_model, inspect_run, post_process, fine_tune

from common.logger import get_logger

log = get_logger(__name__)

JOB_TTL_SECONDS = 2 * 60 * 60

async def create_job(ctx: AppContext) -> JobResponse:
    job_id = uuid4().hex
    job = JobResponse(
        id=job_id,
        status='pending',
        created_at=datetime.now(timezone.utc).isoformat(),
        expires_at=(datetime.now(timezone.utc) + timedelta(seconds=JOB_TTL_SECONDS)).isoformat()
    )

    async with ctx.jobs_lock:
        ctx.jobs[job_id] = job

    return job

async def get_job(ctx: AppContext, job_id: str) -> JobResponse | None:
    async with ctx.jobs_lock:
        job = ctx.jobs[job_id]
        return job if job is not None else None
    
async def run_train_model(ctx: AppContext, job_id: str, cfg: TrainJobRequest) -> None:
    await update_job(ctx, job_id, status='in_progress')
    try:
        params = await get_train_params(ctx, cfg)
        await asyncio.to_thread(train_model, *params)
        async with ctx.jobs_lock:
            ctx.model = None
            ctx.val = None
            ctx.test = None
            ctx.meta = None
            ctx.cached_model_cfg = None
            ctx.cached_dl_cfg = None
            ctx.cached_train_cfg = None

        await update_job(
            ctx,
            job_id,
            status="success",
            status_details="Training finished successfully",
        )
    except Exception as e:
        await update_job(
            ctx,
            job_id,
            status="failed",
            status_details="Training of the model failed",
            error=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc().splitlines()
            )
        )

async def run_inspect_run(ctx: AppContext, job_id: str, cfg: InspectRunJobRequest) -> None:
    await update_job(ctx, job_id, status='in_progress')
    try:
        params = await get_inspect_run_params(ctx, cfg)
        result, ctx_dict = await asyncio.to_thread(inspect_run, *params)
        async with ctx.jobs_lock:
            for k, v in ctx_dict.items():
                if hasattr(ctx, k):
                    setattr(ctx, k, v)
                else:
                    raise ValueError(f'Field {k} does not exist in the AppContext')
        # TODO: update results in job
        await update_job(
            ctx,
            job_id,
            status="success",
            status_details=result
        )
    except Exception as e:
        await update_job(
            ctx,
            job_id,
            status="failed",
            status_details="Inspection failed",
            error=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc().splitlines()
            )
        )

async def run_post_process(ctx: AppContext, job_id: str, cfg: PostProcessingJobRequest) -> None:
    await update_job(ctx, job_id, status='in_progress')
    try:
        params = await get_post_process_params(ctx, cfg)
        result, ctx_dict = await asyncio.to_thread(post_process, *params)
        async with ctx.jobs_lock:
            for k, v in ctx_dict.items():
                if hasattr(ctx, k):
                    setattr(ctx, k, v)
                else:
                    raise ValueError(f'Field {k} does not exist in the AppContext')
        
        await update_job(
            ctx,
            job_id,
            status="success",
            status_details=result
        )
    except Exception as e:
        await update_job(
            ctx,
            job_id,
            status="failed",
            status_details="Post process can't be done",
            error=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc().splitlines()
            )
        )

async def run_fine_tune(ctx: AppContext, job_id: str, cfg: FineTuneJobRequest) -> None:
    await update_job(ctx, job_id, status='in_progress')
    try:
        params = await get_ft_params(ctx, cfg)
        await asyncio.to_thread(fine_tune, *params)
        
        async with ctx.jobs_lock:
            ctx.model = None
            ctx.train = None
            ctx.val = None
            ctx.test = None
            ctx.meta = None
            ctx.cached_model_cfg = None
            ctx.cached_dl_cfg = None
            ctx.cached_train_cfg = None
            ctx.cached_pp_cfg = None
            ctx.cached_run_id = None
        
        await update_job(
            ctx,
            job_id,
            status="success",
            status_details="Fine tuning finished successfully",
        )
    except Exception as e:
        async with ctx.jobs_lock:
            ctx.model = None
            ctx.train = None
            ctx.val = None
            ctx.test = None
            ctx.meta = None
            ctx.cached_model_cfg = None
            ctx.cached_dl_cfg = None
            ctx.cached_train_cfg = None
            ctx.cached_pp_cfg = None
            ctx.cached_run_id = None
        await update_job(
            ctx,
            job_id,
            status="failed",
            status_details="Fine tune failed",
            error=ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc().splitlines()
            )
        )

async def update_job(ctx: AppContext, job_id, **kwargs) -> bool:
    async with ctx.jobs_lock:
        job = ctx.jobs[job_id]
        if not job:
            log.warning(f'There is no job for updating with id: {job_id}')
            return False
        ctx.jobs[job_id] = job.model_copy(update=kwargs)
        return True
    
async def get_train_params(ctx: AppContext, cfg: TrainJobRequest):
    async with ctx.jobs_lock:
        model = ctx.model
        train = ctx.train
        val = ctx.val
        meta = ctx.meta
        dl_cfg = ctx.cached_dl_cfg
        model_cfg = ctx.cached_model_cfg
        train_cfg = ctx.cached_train_cfg
        return (model, train, val, meta, dl_cfg, model_cfg, train_cfg, cfg)

async def get_inspect_run_params(ctx: AppContext, cfg: InspectRunJobRequest):
    async with ctx.jobs_lock:
        client = ctx.mlflow_client
        return (client, cfg)
    
async def get_post_process_params(ctx: AppContext, cfg: PostProcessingJobRequest):
    async with ctx.jobs_lock:
        run_id = ctx.cached_run_id
        model = ctx.model
        val = ctx.val
        meta = ctx.meta
        dl_cfg = ctx.cached_dl_cfg
        model_cfg = ctx.cached_model_cfg
        train_cfg = ctx.cached_train_cfg
        return (model, val, meta, dl_cfg, model_cfg, train_cfg, cfg, run_id)
    
async def get_ft_params(ctx: AppContext, ft_cfg: FineTuneJobRequest):
    async with ctx.jobs_lock:
        run_id = ctx.cached_run_id
        model = ctx.model
        train_dl = ctx.train
        val_dl = ctx.val
        meta = ctx.meta
        dl_cfg = ctx.cached_dl_cfg
        model_cfg = ctx.cached_model_cfg
        train_cfg = ctx.cached_train_cfg
        pp_cfg = ctx.cached_pp_cfg
        return (model, train_dl, val_dl, meta, dl_cfg, model_cfg, train_cfg, pp_cfg, ft_cfg, run_id)