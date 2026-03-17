from pydantic import BaseModel, model_validator
from typing import Optional, Dict

import packages.train_lib.schemas as schemas

class PrepareDatasetJobRequest(BaseModel):
    data_config: schemas.HuggingFaceConfig
    dataset_transforms: schemas.DataTransforms
    batch_size: Optional[int] = 1
    shuffle: Optional[bool] = False
    data_meta_cfg: schemas.DataMetaCfg

class PrepareModelJobRequest(BaseModel):
    model_cfg: schemas.ModelCfg
    model_meta_cfg: Optional[schemas.ModelMetaCfg] = None

    @model_validator(mode='after')
    def validate_graph(self):
        dag = self.model_cfg.dag_cfg
        nodes = dag.nodes
        edges = dag.edges
        node_ids = set(node.id for node in nodes)
        e_node_ids = set([x for t in edges for x in t])
        diff = e_node_ids - node_ids
        if diff:
            raise ValueError(f'DAG configuration error: Invalid nodes  {diff}')
        return self
            
class PrepareTrainJobRequest(BaseModel):
    log_train_metrics: Optional[bool] = False
    device: Optional[str] = 'cpu'
    epochs: int
    num_of_iters: Optional[int] = 1

    optimizer: schemas.OptimizerConfig
    lr_decay: Optional[schemas.LrDecay] = None
    loss_fns: Dict[str, schemas.LossFnCfg]

    metrics: Dict[str, schemas.Metrics]

class PrepareCompleteTrainJobRequest(BaseModel):
    dataset_cfg: PrepareDatasetJobRequest
    model_cfg: PrepareModelJobRequest
    train_cfg: PrepareTrainJobRequest

class LoadRunCfgJobRequest(BaseModel):
    run_id: str

class StartTrainJobRequest(BaseModel):
    exp_name: str
    run_name: str
    model_name: str

class PreparePostProcessingJobRequest(BaseModel):
    new_run_name: str
    post_processors: Dict[str, schemas.PPCfg]

class FinalEvalJobRequest(BaseModel):
    exp_name: str
    run_name: str
    model_name: str