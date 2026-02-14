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
    'data_cfg_rel_path': Path('/data/data_cfg.json'),
    'meta_rel_path': Path('/data/meta.json'),
    'model_cfg_rel_path': Path('/predictor/model_cfg.json'),
    'model_state_rel_path': Path('/predictor/model.pt'),
    'pp_cfg_rel_path': Path('/predictor/pp_cfg.json'),
    'train_cfg_rel_path': Path('/train/train_cfg.json'),
    'error_analysis': Path('/error_analysis/error_analysis.json')
}   

class ArtifactWriter:
    def __init__(self, job_id, run_id):
        self.root = Path(f'/tmp/{job_id}/{run_id}')
        
    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        # TODO: in case job breaks, this can be problematic, for now use writer only in the end of the run
        shutil.rmtree(self.root, ignore_errors=True)
        return False

    def save_data_cfg(self, cfg: dict):
        json_cfg = json.dumps(cfg, indent=2)
        p = self.root / PATHS_MAP['data_cfg_rel_path']
        p.write_text(json_cfg)
    
    def save_meta(self, meta: dict):
        json_meta = json.dumps(meta, indent=2)
        p = self.root / PATHS_MAP['meta_rel_path']
        p.write_text(json_meta)

    def save_model_cfg(self, cfg: dict):
        json_cfg = json.dumps(cfg, indent=2)
        p = self.root / PATHS_MAP['model_cfg_rel_path'] 
        p.write_text(json_cfg)

    def save_model_state(self, model_state: dict):
        p = self.root / PATHS_MAP['model_state_rel_path']
        torch.save(model_state, p)
    
    # TODO: might be None
    def save_post_processor_cfg(self, cfg: dict | None):
        json_cfg = json.dumps(cfg, indent=2)
        p = self.root / PATHS_MAP['pp_cfg_rel_path']
        p.write_text(json_cfg)

    def save_train_cfg(self, cfg: dict):
        json_cfg = json.dumps(cfg, indent=2)
        p = self.root / PATHS_MAP['train_cfg_rel_path']
        p.write_text(json_cfg)

    def ssave_error_analysis(self, error_analysis: dict):
        json_cfg = json.dumps(error_analysis, indent=2)
        p = self.root / PATHS_MAP['error_analysis']
        p.write_text(json_cfg)

    def log_artifacts(self):
        mlflow.log_artifacts(self.root, artifact_path='')

class ArtifactReader:
    def __init__(self, job_id, run_id):
        self.run_id = run_id
        self.root = Path('/artifacts')
        self.cache_dir = Path(f'/cache/{job_id}/{run_id}')

    def __enter__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        return False
    
    def download(self, p):
        return mlflow.artifacts.download_artifacts(
            run_id=self.run_id,
            artifact_path=p,
            dst_path=self.cache_dir
            )

    @staticmethod
    def read_json(cfg_path):
         with open(cfg_path) as f:
            return json.load(f)
    
    def load_data_cfg(self):
        p = self.root / PATHS_MAP['data_cfg_rel_path']
        download_path = self.download(p)
        return self.read_json(download_path)
    
    def load_meta(self):
        p = self.root / PATHS_MAP['meta_rel_path']
        download_path = self.download(p)
        return self.read_json(download_path)

    def load_model_cfg(self):
        p = self.root / PATHS_MAP['model_cfg_rel_path']
        download_path = self.download(p)
        return self.read_json(download_path)
    
    def load_model_state(self):
        p = self.root / PATHS_MAP['model_state_rel_path']
        download_path = self.download(p)
        return torch.load(download_path, map_location="cpu")
    
    def load_post_processor_cfg(self):
        p = self.root / PATHS_MAP['pp_cfg_rel_path']
        download_path = self.download(p)
        return self.read_json(download_path)

    def load_train_cfg(self):
        p = self.root / PATHS_MAP['train_cfg_rel_path']
        download_path = self.download(p)
        return self.read_json(download_path)
    
    def load_error_analysis(self):
        p = self.root / PATHS_MAP['error_analysis']
        download_path = self.download(p)
        return self.read_json(download_path)