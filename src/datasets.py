import networkx as nx
import random
import numpy as np

def sample_random_regular(n, d, seed):
    G = nx.random_regular_graph(d, n, seed=seed)
    return nx.to_numpy_array(G)

def sample_erdos_renyi(n, p, seed):
    G = nx.gnp_random_graph(n, p, seed=seed)
    return nx.to_numpy_array(G)

def sample_dataset_regular_mixed(m: int, n: int, d: int, q: int, seed: int):
    """
    m -> Number of samples
    n -> Number of vertices per graph
    d -> Regularity of graphs
    q -> number of graphs in m that are not regular
    """
    assert q <= m, "Cannot get q more than m"
    rng = random.Random(seed)
    regular = sample_random_regular(n, d, rng.randrange(2**31))
    graphs = []
    for i in range(m):
        if i < (m-q):
            graphs.append(regular.copy())
        else:
            graphs.append(sample_erdos_renyi(n, 0.4, rng.randrange(2**31)))
    return graphs
