from pydantic import BaseModel, model_validator
from typing import Optional, List, Union, Dict

from train_server.schemas.data import HuggingFaceConfig, DataTransforms, DataMetaCfg
from train_server.schemas.models import ModelCfg, ModelMetaCfg
from train_server.schemas.train import OptimizerConfig, LrDecay, LossFnCfg, Metrics
from train_server.schemas.models import PPCfg
# from app_src.schemas.train import FtDatasetCfg, FtLayersCfg, FtTrainCfg

class PrepareDatasetJobRequest(BaseModel):
    data_config: HuggingFaceConfig
    dataset_transforms: DataTransforms
    batch_size: Optional[int] = 1
    shuffle: Optional[bool] = False
    data_meta_cfg: DataMetaCfg

class PrepareModelJobRequest(BaseModel):
    model_config: ModelCfg
    model_meta_cfg: Optional[ModelMetaCfg] = None

    @model_validator(mode='after')
    def validate_graph(self):
        dag = self.model_config.dag
        nodes = dag.nodes
        edges = dag.edges
        node_ids = set(node.id for node in nodes)
        e_node_ids = set([x for t in edges for x in t])
        diff = node_ids - e_node_ids
        if not diff:
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

class InspectJobRequest(BaseModel):
    run_id: str

class PreparePostProcessingJobRequest(BaseModel):
    new_run_name: str
    post_processors: Dict[str, PPCfg]

# class FineTuneJobRequest(BaseModel):
#     new_run_name: str
#     new_ds_cfg: FtDatasetCfg
#     new_layers_cfg: FtLayersCfg
#     new_train_cfg: FtTrainCfg

class FinalEvalJobRequest(BaseModel):
    exp_name: str
    run_name: str
    model_name: str