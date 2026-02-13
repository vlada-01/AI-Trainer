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

    log.info(f'Setting random seeds to {ctx.seed}')
    utils.set_seed(ctx.seed)
    try:
        yield
    finally:
        # TODO: add clean up
        # clean up all runs and jobs
        # ctx.cleanup_task.cancel()
        try:
            # await ctx.cleanup_task
            pass
        except asyncio.CancelledError:
            pass
        # del app.state.ctx

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