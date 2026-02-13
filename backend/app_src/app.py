from fastapi import  FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from mlflow.tracking import  MlflowClient
import asyncio
import os

from app_src.app_ctx import AppContext
from app_src.routes.runs import router as runs_router
from app_src.routes.hf_info import router as hf_info_router


import app_src.utils.app_utils as utils

from common.logger import setup_logging, get_logger

log = get_logger(__name__)

mlflow_public_uri = os.getenv("MLFLOW_PUBLIC_URI", "http://localhost:5000")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
registry_uri = os.getenv("MLFLOW_REGISTRY_URI", "http://localhost:5000")
origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
cleanup_runs_interval = int(os.getenv("CLEANUP_RUNS_INTERVAL", "60"))
cleanup_jobs_interval = int(os.getenv("CLEANUP_JOBS_INTERVAL", "60"))
jobs_ttl = int(os.getenv("JOBS_TTL", "7200"))
runs_inactivity = int(os.getenv("RUNS_INACTIVITY", 1800))

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    log.info('Initializing App context')
    ctx = AppContext(
        seed=42,
        mlflow_client=MlflowClient(
            tracking_uri=tracking_uri,
            registry_uri=registry_uri
        ),
        runs=dict(),
        runs_lock=asyncio.Lock()
    )
 
    app.state.ctx = ctx
    ctx.cleanup_runs_task = asyncio.create_task(utils.cleanup_run_loop(ctx, cleanup_runs_interval))

    log.info(f'Setting random seeds to {ctx.seed}')
    utils.set_seed(ctx.seed)
    try:
        yield
    finally:
        runs = list(ctx.runs.values())
        await asyncio.gather(
        *(run.cancel_cleanup_task() for run in runs),
        return_exceptions=True
        )

        t = getattr(ctx, "cleanup_runs_task", None)
        if t and not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        del app.state.ctx

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(runs_router)
app.include_router(hf_info_router)