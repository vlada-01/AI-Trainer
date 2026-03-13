from enum import Enum
import networkx as nx
import torch.nn as nn

from packages.prepare_model.models.dag_net.nodes import initialize_nodes

from packages.logger.logger import get_logger

log = get_logger(__name__)

def build_dag(cfg, model_meta):
    log.info('Initializing DAG builder')
    nodes_cfg = cfg.nodes
    edges = cfg.edges
    out_keys = cfg.out_keys
    nodes = initialize_nodes(nodes_cfg)
    log.info('Initializing DAGNet')
    dag = DAGNet(nodes, edges, out_keys, model_meta)
    log.info('DAG builder completed successfully')
    return dag

class DAGNet(nn.Module):
    def __init__(self, nodes, edges, out_keys, model_meta):
        super().__init__()

        specs_mapper = model_meta.specs_mapper
        self.out_keys = [specs_mapper(key) for key in out_keys]

        node_ids = nodes.keys()
        graph = nx.DiGraph()
        graph.add_nodes_from([id for id in node_ids])
        graph.add_edges_from([(u, v) for u, v in edges])
        for node in nodes:
            if graph.out_degree(node.id) == 0:
                out_keys = node.get_out_keys()
                mapped = [specs_mapper(k) for k in out_keys]
                node.set_out_keys(mapped)
        
        log.info('Checking graph')
        self.check_graph(graph)
        log.info('Topological sort of the graph')
        self.sorted_ids = self.topological_sort(graph)
        log.debug('Topological sort:\n%s', self.sorted_ids)

        self.nodes = nn.ModuleDict({k: v for k, v in nodes.items()})
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

        out = {k: self.state[k] for k in self.out_keys}
        return out

    def check_graph(self, graph):
        if not graph.is_directed():
            raise ValueError('DAG configuration error: graph is not directed')
        
        if not nx.is_weakly_connected(graph):
            raise ValueError('DAG configuration error: graph is not connected')
        
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError('DAG configuration error: graph is not acyclic')

    def topological_sort(self, graph):
        return list(nx.topological_sort(graph))
    
    



