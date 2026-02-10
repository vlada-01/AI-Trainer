import networkx as nx
import torch.nn as nn
from model_src.models.model_builder import build_layer
from dataclasses import dataclass

from common.logger import get_logger

log = get_logger(__name__)

def construct_dag_model(cfg):
    log.info('Initializing create nodes spec')
    nodes_cfg = cfg.nodes_cfg
    nodes_spec, nodes = create_nodes_spec(nodes_cfg)
    edges = cfg.edges
    dag = DAGNet(nodes_spec, nodes, edges)
    return dag

def create_nodes_spec(nodes_cfg):
    nodes_spec = {}
    nodes = {}
    for cfg in nodes_cfg:
        id = cfg.id
        if cfg.layer is not None:
            cfg_dict = cfg.layer_cfg.model_dump(exclude={'type', 'predefined'})
            layer = build_layer(cfg.layer_cfg.type.value, False, cfg_dict)
            nodes[id] = layer
        else:
            # TODO: check if this is good logic
                nodes[id] = None 
        in_keys = cfg.in_keys
        out_keys = cfg.out_keys
        nodes_spec[id] = NodeSpec(in_keys, out_keys)
    return nodes_spec, nodes

@dataclass
class NodeSpec:
    id: str
    in_keys: list[str]
    out_keys: str

class DAGNet(nn.Module):
    def __init__(self, nodes_spec, nodes, edges):
        super().__init__()
        self.nodes_spec = nodes_spec

        self.nodes = nn.ModuleDict({k: v for k, v in nodes.items()})
        
        graph = nx.DiGraph()
        graph.add_nodes_from([id for id in self.nodes_spec.keys()])
        graph.add_edges_from([(u, v) for u, v in edges.items()])
        
        self.check_graph(graph)
        self.sorted_ids = self.topological_sort()
        
        self.state = {}

    def check_graph(self, graph):
        if not graph.is_directed():
            raise ValueError('Configuration error: graph is not directed')
        
        if not nx.is_weakly_connected(graph):
            raise ValueError('Configuration error: graph is not connected')
        
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError('Configuration error: graph is not acyclic')

    def topological_sort(self, graph):
        return list(nx.topological_sort(graph))
    
    def forward(self, x_dict):
        self.state = x_dict
        for id in self.sorted_ids:
            in_keys = self.nodes_spec.in_keys
            in_kwargs = {k: self.state[k] for k in in_keys}
            out = self.nodes[id](**in_kwargs)
            out = out if isinstance(out, (list, tuple)) else (out, )
            out_keys = self.nodes_spec.out_keys
            if len(out) != len(out_keys):
                raise ValueError(f'Out keys length({len(out_keys)}) and outputs len({len(out)}) does not match for id: {id}')
            self.state.update(dict(zip(out_keys, out)))
        return self.state['final']



