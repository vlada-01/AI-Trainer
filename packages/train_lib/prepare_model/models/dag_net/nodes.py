
import torch.nn as nn
import networkx as nx
import inspect
from abc import ABC, abstractmethod
from enum import Enum

from packages.train_lib.prepare_model.models.dag_net.layers.layers_builder import build_layer, build_layers

from packages.logger.logger import get_logger

log = get_logger(__name__)

class AvailableNodeTypes(str, Enum):
    input = 'input'
    layer = 'layer'
    chain = 'chain'
    component = 'component'

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

class Node(nn.Module, ABC):
    def __init__(self, id, in_keys, out_keys):
        super().__init__()
        self.id = id
        self.in_keys = in_keys
        self.out_keys = out_keys

    def set_out_keys(self, out_keys):
        self.out_keys = out_keys

    def get_id(self):
        return self.id
    
    def get_in_keys(self):
        return self.in_keys
    
    def get_out_keys(self):
        return self.out_keys
    
    @abstractmethod
    def forward(self, *xs):
        pass

class InputNode(Node):
    def __init__(self, id, in_keys, out_keys):
        super().__init__(id, in_keys, out_keys)
        self.exec_node = None
    
    def forward(self, *xs):
        return xs

class LayerNode(Node):
    def __init__(self, id, in_keys, out_keys, layer_cfg):
        super().__init__(id, in_keys, out_keys)
        cfg_dict = layer_cfg.model_dump(exclude={'type'})
        self.exec_node = build_layer(layer_cfg.type.value, cfg_dict)

    def forward(self, *xs):
        return self.exec_node(*xs)
    
class ChainNode(Node):
    def __init__(self, id, in_keys, out_keys, layers_cfg):
        super().__init__(id, in_keys, out_keys)
        self.exec_node = nn.Sequential(*build_layers(layers_cfg))

    def forward(self, *xs):
        return self.exec_node(*xs)

class ComponentNode(Node):
    def __init__(self, id, in_keys, out_keys, nodes, edges):
        super().__init__(id, in_keys, out_keys)

        nn_nodes = self.prepare_nodes(nodes)
        self.exec_node = nn.ModuleDict({k:v for k, v in nn_nodes.items()})

        graph = nx.DiGraph()
        graph.add_nodes_from([id for id in nn_nodes.keys()])
        graph.add_edges_from([(u, v) for u, v in edges])
        
        self.check_graph(graph)
        self.sorted_ids = self.topological_sort(graph)

        self.state = dict()

    def forward(self, *xs):
        self.state = dict(zip(self.in_keys, xs))
        for id in self.sorted_ids:
            in_keys = self.nodes[id].get_in_keys()
            xs = (self.state[k] for k in in_keys)
            out = self.nodes[id](*xs)
            out = out if isinstance(out, (list, tuple)) else (out, )
            out_keys = self.nodes[id].get_out_keys()
            if len(out) != len(out_keys):
                raise ValueError(f'Out keys length({len(out_keys)}) and outputs len({len(out)}) does not match for id: {id}')
            self.state.update(dict(zip(out_keys, out)))
        true_out = (v for k, v in self.state.items() if k in self.out_keys)
        return true_out

    def prepare_nodes(self, nodes):
        nn_nodes = {}
        for node in nodes:
            type = node.type.value
            cls = globals().get(type)
            if not inspect.isclass(cls):
                raise ValueError(f'Unknown ComponentNode type: {type}')
            n = cls(node.model_dump(exclude={'type', 'predefined'}))
            nn_nodes[n.get_id()] = n
        return nn_nodes
    
    def check_graph(self, graph):
        if not graph.is_directed():
            raise ValueError('Configuration error: graph is not directed')
        
        if not nx.is_weakly_connected(graph):
            raise ValueError('Configuration error: graph is not connected')
        
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError('Configuration error: graph is not acyclic')

    def topological_sort(self, graph):
        return list(nx.topological_sort(graph))

NODE_BUILDER_REGISTRY = {
    AvailableNodeTypes.input: InputNode,
    AvailableNodeTypes.layer: LayerNode,
    AvailableNodeTypes.chain: ChainNode,
    AvailableNodeTypes.component: ComponentNode
}