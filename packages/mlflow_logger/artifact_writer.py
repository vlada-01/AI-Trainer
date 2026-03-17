import mlflow
import json
import torch
import shutil
from pathlib import Path

from packages.logger import get_logger

log = get_logger(__name__)

class ArtifactWriter:
    def __init__(self, job_id, run_id):
        self.root = Path(f'/tmp/{job_id}/{run_id}')
        
    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        mlflow.log_artifacts(self.root, artifact_path='')
        shutil.rmtree(self.root, ignore_errors=True)
        return False
    
    # need to be careful to not introduce two files with same name, breaks ArtifactReader logic
    @staticmethod
    def write_text(p: Path, json_dict):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json_dict)

    def log_cfg(self, cfg: dict, rel_path):
        p = self.root / rel_path
        json_cfg = json.dumps(cfg, indent=2)
        self.write_text(p, json_cfg)

    def log_model_state(self, model_state: dict, rel_path):
        p = self.root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_state, p)

    # FIXME: needs to be updated
    def log_meta(self, meta: dict, rel_path):
        json_meta = json.dumps(meta, indent=2)
        p = self.root / rel_path
        self.write_text(p, json_meta)
    
    def save_error_analysis_tables(self, error_analysis_dict: dict):
        p = self.root / 'error_analysis'
        for k, pd_error_table in error_analysis_dict:
            p = p / k / 'error_table.csv'
            p.parent.mkdir(parents=True, exist_ok=True)
            pd_error_table.to_csv(p, mode='a', header=False, index=False)

    def save_error_analysis_extras(self, extras_dict: dict):
        p = self.root / 'error_analysis'
        for k, extras in extras_dict.items():
            p = p  / k / 'extras.txt'
            json_extras = json.dumps(extras, indent=2)
            self.write_text(p, json_extras)