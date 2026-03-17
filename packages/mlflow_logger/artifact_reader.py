import mlflow
import json
import torch
import shutil
from pathlib import Path

from packages.logger import get_logger

log = get_logger(__name__)

class ArtifactReader:
    def __init__(self, job_id, run_id):
        self.run_id = run_id
        self.uri = f'runs:/{run_id}/'
        self.cache_dir = Path(f'/cache/{job_id}/{run_id}')

    def __enter__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        return False
    
    def download(self, uri, path):
        return mlflow.artifacts.download_artifacts(
            artifact_uri=uri,
            dst_path=path
            )
    
    @staticmethod
    def read_json(path):
         with open(path) as f:
            return json.load(f)
         
    def load_cfg(self, rel_path):
        uri = self.uri + rel_path
        download_path = self.download(uri, self.cache_dir)
        return self.read_json(download_path)
    
    def load_model_state(self, rel_path):
        uri = self.uri + rel_path
        download_path = self.download(uri, self.cache_dir)
        return torch.load(download_path, map_location="cpu")
    
    def load_meta(self, rel_path):
        uri = self.uri + rel_path
        download_path = self.download(uri, self.cache_dir)
        return self.read_json(download_path)
    
