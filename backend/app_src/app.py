from fastapi import  FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from mlflow.tracking import  MlflowClient
import asyncio
import os

from app_src.app_ctx import AppContext
from app_src.routes.data import router as data_router
from app_src.routes.models import router as model_router
from app_src.routes.train import router as train_router
from app_src.routes.predict import router as predict_router

from app_src.schemas.models import HistoryResponse, Experiment

import app_src.utils.app_utils as utils

CLEAN_UP_INTERVAL = 60

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
        jobs=dict(),
        jobs_lock=asyncio.Lock()
    )
    ctx.cleanup_task=asyncio.create_task(utils.cleanup_jobs_loop(ctx, CLEAN_UP_INTERVAL))
    app.state.ctx = ctx

    log.info(f'Setting random seeds to {ctx.seed}')
    utils.set_seed(ctx.seed)
    try:
        yield
    finally:
        ctx.cleanup_task.cancel()
        try:
            await ctx.cleanup_task
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
app.include_router(data_router)
app.include_router(model_router)
app.include_router(train_router)
app.include_router(predict_router)

@app.get('/history', response_model=HistoryResponse)
def history():
    try:
        ctx = app.state.ctx
        exps = ctx.mlflow_client.search_experiments()
        active_exps = [e for e in exps if e.lifecycle_stage == 'active']
        list_of_exps = [
            Experiment(
                name=e.name,
                url=f'{mlflow_public_uri}/#/experiments/{e.experiment_id}'
            )
            for e in active_exps
        ]
        return HistoryResponse(
            exps=list_of_exps
        )
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error_type": type(e).__name__,
                "error_message":str(e)
                }
            ) 