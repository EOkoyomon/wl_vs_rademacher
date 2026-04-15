import networkx as nx
import random
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import from_networkx
import os
from torch_geometric.data import InMemoryDataset
import csv

class RandomLabelMemorizationDataset(InMemoryDataset):
    def __init__(self, root, num_graphs=10, nodes=30, degree=3, 
                 regime='all', K=0, rho=1.0, perturbation_type='edge_rewire', 
                 wl_iter=3, seed=42, transform=None):

        self.num_graphs = num_graphs
        self.nodes = nodes
        self.degree = degree
        self.regime = regime # 'all' or 'fraction'
        self.K = K # Number of local graph perturbations
        self.rho = rho # Fraction of samples to perturb
        self.perturbation_type = perturbation_type
        self.wl_iter = wl_iter
        self.seed = seed

        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if self.regime == 'all':
            return [f'S_all_K{self.K}_N{self.num_graphs}_n{self.nodes}_deg{self.degree}_P-{self.perturbation_type}_seed{self.seed}.pt']
        elif self.regime == 'fraction':
            return [f'S_rho{self.rho:.2f}__K{self.K}_N{self.num_graphs}_n{self.nodes}_deg{self.degree}_P-{self.perturbation_type}_seed{self.seed}.pt']

    def perturb_graph(self, G_original, num_steps):
        """
        Given an original graph G_original, perform num_steps random local perturbations.
        The type of perturbation is determined by self.perturbation_type, which can be:
        - edge_rewire: randomly select an edge (u, v), remove it, and add a new edge (u, w) where w is a random node not connected to u.
        - edge_add: randomly select two unconnected nodes (u, v) and add an edge between them.
        - edge_remove: randomly select an existing edge (u, v) and remove it.
        - node_remove: randomly select a node n and remove it along with its edges"""
        G = G_original.copy()
        if num_steps == 0:
            return G
        

        for _ in range(num_steps):
            if self.perturbation_type == 'edge_rewire':
                edges = list(G.edges())
                nodes = list(G.nodes())

                u, v = random.choice(edges)
                G.remove_edge(u, v)
                if (u, v) in edges: edges.remove((u, v)) # Update edge list after removal
                if (v, u) in edges: edges.remove((v, u))
                
                while True:
                    w = random.choice(nodes) # not reproducible
                    if w != u and not G.has_edge(u, w): # avoid self-loops and existing edges
                        G.add_edge(u, w)
                        edges.append((u, w)) # update edge list
                        break
            
            elif self.perturbation_type == 'edge_add':
                # Add an edge between two previously unconnected nodes
                non_edges = list(nx.non_edges(G))
                if not non_edges: break # No more non-edges to add
                u, v = random.choice(non_edges)
                G.add_edge(u, v)
                
            elif self.perturbation_type == 'edge_remove':
                # Remove an existing edge
                edges = list(G.edges())
                if not edges: break # No edges left to remove
                u, v = random.choice(edges)
                G.remove_edge(u, v)
                
            elif self.perturbation_type == 'node_remove':
                # Remove a random node (and consequently its edges)
                nodes = list(G.nodes())
                if not nodes: break
                n = random.choice(nodes)
                G.remove_node(n)
                
            else:
                raise ValueError(f"Unknown perturbation: {self.perturbation_type}")

        return G


    def process(self):
        data_list = []
        wl_hashes = []
        
        # G_original = nx.random_regular_graph(self.degree, self.nodes) #N = 20 already >500.000 not isomorphic graphs!
        # Other options: nx.cycle_graph(N), watts_strogatz_graph(N, k, p=0) # Regular ring graph with N nodes and degree k (it works only with even k!) 
        
        print(f"Dataset Generation: {self.processed_file_names[0]} ---")
        # print("-" * 60)
        # print(f"{'Rewires (K)':<15} | {'Unique Classes (WL)':<20} | {'Note'}")
        # print("-" * 60)
        random.seed(self.seed)

        base_graphs = [nx.random_regular_graph(self.degree, self.nodes, seed=self.seed+i) 
                       for i in range(self.num_graphs)]
        
        targets = [-1] * (self.num_graphs // 2) + [1] * (self.num_graphs - self.num_graphs // 2) # Assign random labels (-1 or 1) to the graphs, ensuring a balanced distribution
        random.shuffle(targets)

        if self.regime == 'all':
            sample_to_perturb = self.num_graphs
        elif self.regime == 'fraction':
            sample_to_perturb = int(self.rho * self.num_graphs)

        for i in range(self.num_graphs):
            if i < sample_to_perturb: #if sample_to_perturb == self.num_graphs, all samples are perturbed
                G_p = self.perturb_graph(base_graphs[i], self.K)
            else: # Perturb only a fraction of the samples and ensures monotonicity for larger \rho values
                G_p = self.perturb_graph(base_graphs[i], 0) # No perturbation

            # computing the wl-hashing for the perturbed graph
            h = nx.weisfeiler_lehman_graph_hash(G_p, iterations=self.wl_iter)
            wl_hashes.append(h)
            
            data = from_networkx(G_p)
            data.x = torch.ones((data.num_nodes, 1)) # Feature all 1
            data.y = torch.tensor([targets[i]], dtype=torch.long) # Label 
            
            # Metadata saved in the graph data object
            data.K = torch.tensor([self.K], dtype=torch.long) 
            data.rho = torch.tensor([self.rho], dtype=torch.float)
            data.wl_hash = h # Save the WL hash as a string attribute (not a tensor)
            
            data_list.append(data)
            
        # print wl statistics for this K
        unique_classes = len(set(wl_hashes))
        print(f"-> Regime: {self.regime.upper()} | K: {self.K} | Rho: {self.rho} | Type: {self.perturbation_type}")
        print(f"-> WL Equivalence Classes (p): {unique_classes} / {self.num_graphs}")
       
        print("-" * 60)

        csv_file = os.path.join(self.root, "wl_statistics.csv")
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow(["Num_Graphs", "Nodes", "Degree","Regime", "K", "Rho", "Perturbation_Type", "WL_Classes",])
            
            
            writer.writerow([
                self.num_graphs,
                self.nodes, 
                self.degree,
                self.regime, 
                self.K, 
                self.rho, 
                self.perturbation_type, 
                unique_classes
            ])
        print("Saving dataset...")
        self.save(data_list, self.processed_paths[0])
        print("Dataset saved successfully!")


if __name__ == "__main__":
    for k in [0, 1, 2, 3, 4, 5]:
        dataset = RandomLabelMemorizationDataset(root='./data', regime='all', K=k)

    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        dataset = RandomLabelMemorizationDataset(root='./data', regime='fraction', K=3, rho=r)
