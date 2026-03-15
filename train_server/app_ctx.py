from dataclasses import dataclass
from typing import Dict, Optional
import asyncio
import mlflow

from packages.server_lib.runs.run_ctx import RunContext

@dataclass
class AppContext:
    seed: int
    mlflow_client: mlflow.MlflowClient

    runs: Dict[str, RunContext]
    runs_lock: asyncio.Lock

    cleanup_runs_task: Optional[asyncio.Task] = None