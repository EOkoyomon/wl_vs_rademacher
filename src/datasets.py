import networkx as nx
import random
import numpy as np

def sample_random_regular(n, d, seed):
    G = nx.random_regular_graph(d, n, seed=seed)
    return nx.to_numpy_array(G)

def sample_erdos_renyi(n, p, seed):
    G = nx.gnp_random_graph(n, p, seed=seed)
    return nx.to_numpy_array(G)

def sample_dataset_regular_mixed(m: int, n: int, d: int, q: int, seed: int, prop: float = None):
    """
    m -> Number of samples
    n -> Number of vertices per graph
    d -> Regularity of graphs
    q -> number of distinct WL-rich graph classes
    prop -> proportion of m allocated to each q 
    """
    assert q <= m, "Cannot get q more than m"
    if prop is not None:
        assert prop * q < 1.0, "No space for all q"
    rng = random.Random(seed)
    regular = sample_random_regular(n, d, rng.randrange(2**31))
    n_per_q_class = max(1, round(prop * m)) if prop is not None else 1

    graphs = []
    for _ in range(q):
        q_class = sample_erdos_renyi(n, 0.4, rng.randrange(2**31))
        graphs.extend(q_class.copy() for _ in range(n_per_q_class))

    n_regular = m - len(graphs)
    assert n_regular > 0, "No space for regular graphs"
    graphs.extend(regular.copy() for _ in range(n_regular))
    return graphs
