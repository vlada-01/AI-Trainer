from enum import Enum
import networkx as nx
import torch.nn as nn

from model_src.models.model_types.nodes import InputNode, LayerNode, ChainNode, ComponentNode

from common.logger import get_logger

log = get_logger(__name__)

class AvailableNodeTypes(str, Enum):
    input = 'input'
    layer = 'layer'
    chain = 'chain'
    component = 'component'

NODE_BUILDER_REGISTRY = {
    AvailableNodeTypes.input: InputNode,
    AvailableNodeTypes.layer: LayerNode,
    AvailableNodeTypes.chain: ChainNode,
    AvailableNodeTypes.component: ComponentNode
}

def dag_builder(cfg):
    log.info('Initializing DAG builder')
    nodes_cfg = cfg.nodes
    dag_cfg = cfg.dag
    nodes = initialize_nodes(nodes_cfg)
    log.info('Initializing DAGNet')
    dag = DAGNet(dag_cfg, nodes)
    log.info('DAG builder completed successfully')
    return dag

def initialize_nodes(nodes_cfg):
    nodes = {}
    for node_cfg in nodes_cfg:
        node_type = node_cfg.type
        cfg_dict = {k: getattr(node_cfg, k) for k in node_cfg.model_fields if k != 'type'}
        node = build_node(node_type, cfg_dict)
        nodes[node.id] = node
    log.debug('Assembled nodes:\n%s', {k: type(v) for k, v in nodes.items()})
    return nodes

def build_node(type, cfg_dict):
    if type not in NODE_BUILDER_REGISTRY:
        raise ValueError(f'Type "{type}" is not known node type')
    log.info(f'Initializing node of type: {type}')
    return NODE_BUILDER_REGISTRY[type](**cfg_dict)

class DAGNet(nn.Module):
    def __init__(self, dag_cfg, nodes):
        super().__init__()

        self.out_key = dag_cfg.out_key

        self.nodes = nn.ModuleDict({k: v for k, v in nodes.items()})
        
        node_ids = dag_cfg.node_ids
        edges = dag_cfg.edges
        graph = nx.DiGraph()
        graph.add_nodes_from([id for id in node_ids])
        graph.add_edges_from([(u, v) for u, v in edges])
        
        log.info('Checking graph')
        self.check_graph(graph)
        log.info('Topological sort of the graph')
        self.sorted_ids = self.topological_sort(graph)
        log.debug('Topological sort:\n%s', self.sorted_ids)
        
        self.state = dict()

    def forward(self, x_dict):
        self.state = x_dict
        for id in self.sorted_ids:
            in_keys = self.nodes[id].get_in_keys()
            xs = (self.state[k] for k in in_keys)
            out = self.nodes[id](*xs)
            out = out if isinstance(out, (list, tuple)) else (out, )
            out_keys = self.nodes[id].get_out_keys()
            if len(out) != len(out_keys):
                raise ValueError(f'Out keys length({len(out_keys)}) and outputs len({len(out)}) does not match for id: {id}')
            self.state.update(dict(zip(out_keys, out)))
        return self.state[self.out_key]

    def check_graph(self, graph):
        if not graph.is_directed():
            raise ValueError('DAG configuration error: graph is not directed')
        
        if not nx.is_weakly_connected(graph):
            raise ValueError('DAG configuration error: graph is not connected')
        
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError('DAG configuration error: graph is not acyclic')

    def topological_sort(self, graph):
        return list(nx.topological_sort(graph))
    
    



