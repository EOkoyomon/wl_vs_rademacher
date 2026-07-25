import networkx as nx
from torch_geometric.utils import to_networkx
import torch
import math

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

def wl_colors(graphs, iterations):
    g_hashes = []
    for g in graphs:
        G = nx.from_numpy_array(g)
        g_hash = nx.algorithms.weisfeiler_lehman_graph_hash(G, iterations=iterations)
        g_hashes.append(g_hash)
    return g_hashes

    
def partition_by_invariant(invariants):
    groups = {}
    for i, t in enumerate(invariants):
        groups.setdefault(t, []).append(i)
    return groups


def prop2_bounds(mu_list, mu_counts, m):
    upper = sum(math.sqrt(mu / m) for mu in mu_list)
    lower = sum(math.sqrt(mu / (2 * m)) for mu in mu_list)
    total = 0.0

    for mu in mu_counts:
        if mu == 0:
            continue
        e = sum(math.comb(mu, k) * abs(2 * k - mu) for k in range(mu + 1)) / (2 ** mu)
        total += e

    return lower, upper, total/m
