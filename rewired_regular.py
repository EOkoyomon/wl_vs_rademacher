import networkx as nx
import random
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import from_networkx
import os
from torch_geometric.data import InMemoryDataset

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
import networkx as nx
import random
import os

class RewiredRegularGraphsDataset(InMemoryDataset):
    def __init__(self, root, num_graphs=10, nodes=30, degree=3, max_rewires=5, wl_iter=3, transform=None):
        self.num_graphs = num_graphs
        self.nodes = nodes
        self.degree = degree
        self.max_rewires = max_rewires
        self.wl_iter = wl_iter
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):

        return [f'rewired_samples_{self.num_graphs}_Nodes_{self.nodes}_Deg_{self.degree}_maxK_{self.max_rewires}.pt']
    def perturb_graph(self, G_original, num_rewires):
        """
        Given an original graph G_original, perform num_rewires random edge rewires.
        Each rewire consists of removing one edge and adding a new edge between
        two previously unconnected nodes.
        """
        G = G_original.copy()
        if num_rewires == 0:
            return G
        edges = list(G.edges())
        nodes = list(G.nodes())
        for _ in range(num_rewires):
            u, v = random.choice(edges)
            G.remove_edge(u, v)
            if (u, v) in edges: edges.remove((u, v))
            if (v, u) in edges: edges.remove((v, u))
            
            while True:
                w = random.choice(nodes) # not reproducible
                if w != u and not G.has_edge(u, w): # avoid self-loops and existing edges
                    G.add_edge(u, w)
                    edges.append((u, w)) # update edge list
                    break
        return G


    def process(self):
        data_list = []
        G_original = nx.random_regular_graph(self.degree, self.nodes) #N = 20 already >500.000 not isomorphic graphs!
        # Other options: nx.cycle_graph(N), watts_strogatz_graph(N, k, p=0) # Regular ring graph with N nodes and degree k (it works only with even k!) 
        
        print(f"Generazione Dataset: Nodes={self.nodes}, Deg={self.degree}, Samples={self.num_graphs}, WL_iter={self.wl_iter}")
        print("-" * 60)
        print(f"{'Rewires (K)':<15} | {'Unique Classes (WL)':<20} | {'Note'}")
        print("-" * 60)

        for K in range(self.max_rewires + 1):
            wl_hashes = []
            
            for _ in range(self.num_graphs):
                G_p = self.perturb_graph(G_original, K)
                
                # computing the wl-hashing for the perturbed graph
                h = nx.weisfeiler_lehman_graph_hash(G_p, iterations=self.wl_iter)
                wl_hashes.append(h)
                
                data = from_networkx(G_p)
                data.x = torch.ones((data.num_nodes, 1)) # Feature all 1
                data.y = torch.tensor([0], dtype=torch.long) # Label 
                
                # Metadata saved in the graph data object
                data.K = torch.tensor([K], dtype=torch.long) #rewires
                data.wl_hash = h # Save the WL hash as a string attribute (not a tensor)
                
                data_list.append(data)
            
            # print wl statistics for this K
            unique_classes = len(set(wl_hashes))
            note = "ALL isomorphic" if unique_classes == 1 else ""
            if unique_classes == self.num_graphs: 
                note = "ALL Distinct"
            
            print(f"{K:<15} | {unique_classes:<20} | {note}")

        print("-" * 60)
        print("Saving dataset...")
        self.save(data_list, self.processed_paths[0])
        print("Dataset saved successfully!")


if __name__ == "__main__":
    dataset = RewiredRegularGraphsDataset(root='./data')
