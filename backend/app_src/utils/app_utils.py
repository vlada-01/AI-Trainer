import torch
import random
import numpy as np
import asyncio

from app_src.app_ctx import AppContext

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

async def cleanup_run_loop(ctx: AppContext, interval_seconds: int):
    while True:
        await asyncio.sleep(interval_seconds)

        async with ctx.runs_lock:
            expired_ids = [
                run_id
                for run_id, run in ctx.runs.items()
                if await run.is_cleanable()
            ]
            for run_id in expired_ids:
                del ctx.runs[run_id]
