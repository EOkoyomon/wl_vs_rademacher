from typing import List
import networkx as nx
from torch_geometric.utils import to_networkx
from collections import defaultdict
import torch

# Define transform: convert node labels to features if present
class OneHotNodeLabel:
    def __call__(self, data):
        if hasattr(data, 'node_label'):
            num_classes = int(data.node_label.max().item()) + 1
            data.x = torch.nn.functional.one_hot(data.node_label, num_classes=num_classes).float()
        elif data.x is None:
            # Fallback: constant features
            data.x = torch.ones((data.num_nodes, 1))
        return data

class CalculateWLColors:
    def __init__(self, list_num_iterations: List[int] = [3]):
        self.list_num_iterations = list_num_iterations
    def __call__(self, data):
        hash_dict = defaultdict(str)
        for num_iters in self.list_num_iterations:
            G = to_networkx(data)
            g_hash = nx.algorithms.weisfeiler_lehman_graph_hash(G, iterations=num_iters+1)
            hash_dict[num_iters] = g_hash
        for k,v in hash_dict.items():
            data[f"wl_hash_{k}"] = v
        return data



