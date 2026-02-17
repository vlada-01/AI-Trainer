import os
import mlflow
import json
import torch
import shutil
from pathlib import Path

from common.logger import get_logger

# TODO: add logs later
log = get_logger(__name__)

PATHS_MAP = {
    'data_cfg_rel_path': Path('data/data_cfg.json'),
    'meta_rel_path': Path('data/meta.json'),
    'model_cfg_rel_path': Path('predictor/model_cfg.json'),
    'model_state_rel_path': Path('predictor/model.pt'),
    'pp_cfg_rel_path': Path('predictor/pp_cfg.json'),
    'train_cfg_rel_path': Path('train/train_cfg.json'),
    'error_analysis_rel_path': Path('error_analysis/error_analysis.json')
}   

class ArtifactWriter:
    def __init__(self, job_id, run_id):
        self.root = Path(f'/tmp/{job_id}/{run_id}')
        
    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        shutil.rmtree(self.root, ignore_errors=True)
        return False
    
    # TODO: need to be careful to not introduce two files with same name, breaks ArtifactReader logic
    @staticmethod
    def write_text(p: Path, json_cfg):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json_cfg)

    def save_data_cfg(self, cfg: dict):
        json_cfg = json.dumps(cfg, indent=2)
        p = self.root / PATHS_MAP['data_cfg_rel_path']
        self.write_text(p, json_cfg)
    
    def save_meta(self, meta: dict):
        json_meta = json.dumps(meta, indent=2)
        p = self.root / PATHS_MAP['meta_rel_path']
        self.write_text(p, json_meta)

    def save_model_cfg(self, cfg: dict):
        json_cfg = json.dumps(cfg, indent=2)
        p = self.root / PATHS_MAP['model_cfg_rel_path']
        self.write_text(p, json_cfg)

    def save_model_state(self, model_state: dict):
        p = self.root / PATHS_MAP['model_state_rel_path']
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_state, p)
    
    # TODO: might be None
    def save_post_processor_cfg(self, cfg: dict | None):
        json_cfg = json.dumps(cfg, indent=2)
        p = self.root / PATHS_MAP['pp_cfg_rel_path']
        self.write_text(p, json_cfg)

    def save_train_cfg(self, cfg: dict):
        json_cfg = json.dumps(cfg, indent=2)
        p = self.root / PATHS_MAP['train_cfg_rel_path']
        self.write_text(p, json_cfg)

    def save_error_analysis(self, error_analysis: dict):
        json_cfg = json.dumps(error_analysis, indent=2)
        p = self.root / PATHS_MAP['error_analysis_rel_path']
        self.write_text(p, json_cfg)

    def log_artifacts(self):
        mlflow.log_artifacts(self.root, artifact_path='')

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
    
    def download(self, uri):
        return mlflow.artifacts.download_artifacts(
            artifact_uri=uri,
            dst_path=self.cache_dir
            )

    @staticmethod
    def read_json(cfg_path):
         with open(cfg_path) as f:
            return json.load(f)
    
    def load_data_cfg(self):
        uri = self.uri + str(PATHS_MAP['data_cfg_rel_path'])
        download_path = self.download(uri)
        return self.read_json(download_path)
    
    def load_meta(self):
        uri = self.uri + str(PATHS_MAP['meta_rel_path'])
        download_path = self.download(uri)
        return self.read_json(download_path)

    def load_model_cfg(self):
        uri = self.uri + str(PATHS_MAP['model_cfg_rel_path'])
        download_path = self.download(uri)
        return self.read_json(download_path)
    
    def load_model_state(self):
        uri = self.uri + str(PATHS_MAP['model_state_rel_path'])
        download_path = self.download(uri)
        return torch.load(download_path, map_location="cpu")
    
    def load_post_processor_cfg(self):
        uri = self.uri + str(PATHS_MAP['pp_cfg_rel_path'])
        download_path = self.download(uri)
        return self.read_json(download_path)

    def load_train_cfg(self):
        uri = self.uri + str(PATHS_MAP['train_cfg_rel_path'])
        download_path = self.download(uri)
        return self.read_json(download_path)
    
    def load_error_analysis(self):
        uri = self.uri + str(PATHS_MAP['error_analysis_rel_path'])
        download_path = self.download(uri)
        return self.read_json(download_path)