from dataclasses import dataclass
from typing import Dict
import asyncio
import mlflow

from app_src.run_ctx import RunContext

@dataclass
class AppContext:
    seed: int
    mlflow_client: mlflow.MlflowClient

    runs: Dict[str, RunContext]
    runs_lock: asyncio.Lock