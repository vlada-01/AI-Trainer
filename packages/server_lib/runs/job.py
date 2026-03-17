import os
from uuid import uuid4
from datetime import datetime, timezone, timedelta
from enum import Enum

jobs_ttl = int(os.getenv("JOBS_TTL", "7200"))

class JobStatus(str, Enum):
    pending = 'pending'
    in_progress = 'in_progress'
    success = 'success'
    failed = 'failed'

class Job:
    def __init__(self, state_code):
        self.id = uuid4().hex
        self.state_code = state_code
        self.status = JobStatus.pending
        self.status_details = None
        self.created_at = datetime.now(timezone.utc)
        self.expires_at = (datetime.now(timezone.utc) + timedelta(seconds=jobs_ttl))

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f'Job does not have field named: {k}')
            setattr(self, k, v)

    def expired(self, now):
        return self.expires_at <= now
    
    def get_id(self):
        return self.id

    def to_dict(self):
        return {
            'job_id': self.id,
            'job_type': self.state_code,
            'status': self.status,
            'status_details': self.status_details,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
        }