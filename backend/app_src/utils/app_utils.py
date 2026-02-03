import torch
import random
import numpy as np
import asyncio
from datetime import datetime, timezone

from app_src.app_ctx import AppContext

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

async def cleanup_jobs_loop(ctx: AppContext, interval_seconds: int):
    while True:
        await asyncio.sleep(interval_seconds)
        now = datetime.now(timezone.utc)

        async with ctx.jobs_lock:
            expired_ids = [
                job_id
                for job_id, job in ctx.jobs.items()
                if job.expires_at and datetime.fromisoformat(job.expires_at) <= now
            ]
            for job_id in expired_ids:
                del ctx.jobs[job_id]