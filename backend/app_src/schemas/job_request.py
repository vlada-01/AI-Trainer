from pydantic import BaseModel, model_validator
from typing import Optional, List, Union

from app_src.schemas.data import SklearnConfig, HuggingFaceConfig, DataTransforms
from app_src.schemas.models import DAGCfg, NodeCfg
from app_src.schemas.train import TrainCfg
from app_src.schemas.train import FtDatasetCfg, FtLayersCfg, FtTrainCfg
from app_src.schemas.train import Calibration, GlobalThreshold

class PrepareDatasetJobRequest(BaseModel):
    data_config: Union[SklearnConfig, HuggingFaceConfig]
    dataset_transforms: DataTransforms
    batch_size: Optional[int] = 1
    shuffle: Optional[bool] = False

class PrepareModelJobRequest(BaseModel):
    nodes: List[NodeCfg]
    dag: DAGCfg

    @model_validator(mode='after')
    def validate_graph(self):
        node_ids = set([id for id in self.dag.node_ids])

        # node ids must be unique
        if len(node_ids) != len(self.dag.node_ids):
            raise ValueError(f'DAG configuration error: contains duplicates') 
        
        # all nodes must be also present in dag.nodes
        for node in self.nodes:
            if node.id not in node_ids:
                raise ValueError(f'DAG configuration error: Invalid node {node.id}')
            
        # all edges must be present in dag.nodes
        for u, v in self.dag.edges:
            if u not in node_ids or v not in node_ids:
                raise ValueError(f'DAG configuration error: Invalid edge ({u},{v})')
            
class PrepareTrainJobRequest(BaseModel):
    exp_name: str
    run_name: str
    model_name: str
    log_train_metrics: Optional[bool] = False
    train_cfg: TrainCfg

class PrepareCompleteTrainJobRequest(BaseModel):
    dataset_cfg: PrepareDatasetJobRequest
    model_cfg: PrepareModelJobRequest
    train_cfg: PrepareTrainJobRequest

class StartTrainJobRequest(BaseModel):
    name: None

class InspectJobRequest(BaseModel):
    run_id: str

class PreparePostProcessingJobRequest(BaseModel):
    new_run_name: str
    post_processors: List[Union[
        Calibration,
        GlobalThreshold
    ]]

class FineTuneJobRequest(BaseModel):
    new_run_name: str
    new_ds_cfg: FtDatasetCfg
    new_layers_cfg: FtLayersCfg
    new_train_cfg: FtTrainCfg

class PredictJobRequest(BaseModel):
    run_id: str