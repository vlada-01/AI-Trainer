from pydantic import BaseModel, model_validator
from typing import Optional, Dict

from packages.train_lib.schemas.data import HuggingFaceConfig, DataTransforms, DataMetaCfg
from packages.train_lib.schemas.models import ModelCfg, ModelMetaCfg
from packages.train_lib.schemas.train import OptimizerConfig, LrDecay, LossFnCfg, Metrics
from packages.train_lib.schemas.models import PPCfg
# from app_src.schemas.train import FtDatasetCfg, FtLayersCfg, FtTrainCfg

class PrepareDatasetJobRequest(BaseModel):
    data_config: HuggingFaceConfig
    dataset_transforms: DataTransforms
    batch_size: Optional[int] = 1
    shuffle: Optional[bool] = False
    data_meta_cfg: DataMetaCfg

class PrepareModelJobRequest(BaseModel):
    model_cfg: ModelCfg
    model_meta_cfg: Optional[ModelMetaCfg] = None

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

    optimizer: OptimizerConfig
    lr_decay: Optional[LrDecay] = None
    loss_fns: Dict[str, LossFnCfg]

    metrics: Dict[str, Metrics]

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
    post_processors: Dict[str, PPCfg]

class FinalEvalJobRequest(BaseModel):
    exp_name: str
    run_name: str
    model_name: str