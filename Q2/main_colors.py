import csv
import os.path as osp

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Linear, global_mean_pool, GraphConv
from tqdm import tqdm

batch_size = 128
# num_layers = [0, 1, 2, 3, 4, 5, 6]
num_layers = [0]
lr = 0.001
epochs = 5
dataset_name_list = ["MCF-7", "MCF-7H", "MUTAGENICITY", "NCI1", "NCI109"] # "ENZYMES" is not a binary classification case, even though it was included in the Garg et al. 2020 paper.
dataset_name_list = ["NCI1"] # For testing.
num_reps = 1
hd = 64

color_counts = [
    [3, 231, 10416, 15208, 16029, 16450, 16722, 16895, 17026]
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nc):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(nc):
            self.convs.append(GraphConv(in_channels, hidden_channels, aggr='add', bias=True))
            in_channels = hidden_channels

        if nc != 0:
            self.readout = Linear(hidden_channels, out_channels)
        else:
            self.readout = Linear(in_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
        x = torch.sigmoid(self.readout(x))

        return global_mean_pool(x, batch)


for d, dataset_name in enumerate(dataset_name_list):
    print("Loading dataset", dataset_name, end='... ')
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
    dataset = TUDataset(path, name=dataset_name).shuffle()
    print("done")

    colors = sns.color_palette()

    raw_data = []
    table_data = []

    diffs = []
    diffs_std = []

    for l in num_layers:
        print("Number of layers:", l)
        table_data.append([])
        for it in range(num_reps):
            print("Repetition", it)

            dataset.shuffle()

            train_dataset = dataset[len(dataset) // 10:]
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

            test_dataset = dataset[:len(dataset) // 10]
            test_loader = DataLoader(test_dataset, batch_size)

            model = Net(dataset.num_features, hd, 1, l).to(device) # Binary

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            def population_risk(pred, labels):
                # probs: tensor of shape [batch_size], values in [0, 1]
                # labels: tensor of shape [batch_size], values 0 or 1
                return (labels * (2 * pred - 1) + (1 - labels) * (1 - 2 * pred)).mean()

            def train():
                model.train()

                total_loss = 0
                for data in train_loader:
                    data = data.to(device)
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index, data.batch).squeeze(-1)
                    labels = data.y.view(-1).float() # Convert data.y to float from int
                    loss = population_risk(out, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss) * data.num_graphs
                return total_loss / len(train_loader.dataset)


            @torch.no_grad()
            def test(loader):
                model.eval()

                total_correct = 0
                for data in loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.batch).squeeze(-1)
                    probs = torch.sigmoid(out) # convert to [0, 1] again just to separate for test
                    pred = (probs > 0.5).long() # threshold at 0.5
                    total_correct += int((pred == data.y).sum())
                return total_correct / len(loader.dataset)
            
            def max_l2_norms_conv_weights(model):
                w1_norms = [0] # For the case with no layers, the weights would be 0.
                w2_norms = [0] # For the case with no layers, the weights would be 0.
                
                for conv in model.convs:
                    print("In a conv")
                    # Assuming conv.lin and conv.root are the weight layers
                    # conv.lin is used for neighbors (W2), conv.root is used for the root/self node (W1)
                    W2 = conv.lin_rel.weight
                    W1 = conv.lin_root.weight
                    
                    w2_norm = torch.norm(W2, p=2)
                    w1_norm = torch.norm(W1, p=2)
                    
                    w2_norms.append(w2_norm.item())
                    w1_norms.append(w1_norm.item())
                
                return max(w1_norms), max(w2_norms)
            
            def max_l2_norm_inputs(dataset):
                max_norm = 0.0
                for data in dataset:
                    # data.x is of shape [num_nodes_in_graph, r]
                    norm = torch.norm(data.x, p=2)  # Frobenius norm
                    if norm.item() > max_norm:
                        max_norm = norm.item()
                return max_norm
            
            def max_branching_factor(dataset):
                max_degree = 0
                for data in dataset:
                    # data.edge_index is of shape [2, num_edges]
                    row = data.edge_index[0]  # source nodes of edges
                    deg = torch.bincount(row, minlength=data.num_nodes)  # degree of each node
                    max_deg_in_graph = deg.max().item()
                    if max_deg_in_graph > max_degree:
                        max_degree = max_deg_in_graph
                return max_degree
            
            @torch.no_grad()
            def calculate_rademacher_complexity(model, dataset, hidden_dim, gnn_depth):
                """
                Calculates the upper bound of the rademacher complexity for the GNN model and dataset.
                Based on Garg et al., 2020
                """
                # r is the dimension of the embedding
                r = hidden_dim

                # d is the branching factor (i.e. max number of neighbours for any node)
                d = max_branching_factor(dataset)

                # m is the sample size
                m = dataset.len()

                # L is the depth of the GNN
                L = gnn_depth

                # C_rho is the Lipschitz constant of the transform (rho) of the locally-invariant neighbourhood aggregations. The paper does not indicate nonlinear but the survey paper says nonlinear.
                C_rho = 1.0 # Can use ReLU, but using identity in the code by default
                
                # C_g is the Lipschitz constant of the nonlinear transform (g) applied to each neighbour before aggregation
                C_g = 1.0 # Can use ReLU, but using identity in the code by default
                
                # C_phi is the Lipschitz constant of the non-linear transform (phi) applied at the end of the embedding update
                C_phi = 1.0 # Tanh

                # b is the L_infinity norm of phi.
                b = 1.0 # Because Tanh

                # B1 is the upper bound of the L2 norm of the weight matrix that is applied to the main node embedding (x_v) before combination with its neighbours. The survey paper applies it to h_v rather than x_v and bounds this.
                # B2 is the upper bound of the L2 norm of the weight matrix that is applied to the aggregated neighbours, after the rho transformation.
                B1, B2 = max_l2_norms_conv_weights(model)

                # B_x is the upper bound of the L2 norm of the feature vector x_v
                B_x = max_l2_norm_inputs(dataset)

                # B_beta is the upper bound of the L2 norm of weight matrix, Beta, of the readout function, used to apply a binary classifier by converting from feature dim to 1 dim.
                B_beta = torch.norm(model.readout.weight, p=2)

                # The margin for the margin loss.
                gamma = 1.0
                
                # Calculate the rademacher complexity upper bound
                C = C_rho * C_g * C_phi * B2
                if C * d == 1:
                    M = C_phi * L
                else:
                    M = C_phi * ((C * d) ** L - 1) / (C * d - 1)

                R = C_rho * C_g * d * min(b * torch.sqrt(torch.tensor(r)).item(), B1 * B_x * M)
                Z = C_phi * B1 * B_x + C_phi * B2 * R
                Q = 24 * B_beta * torch.sqrt(torch.tensor(m)).item() * max(Z, M * torch.sqrt(torch.tensor(r)).item() * max(B_x * B1, R * B2))

                first_term = 4 / (gamma * m)
                second_term = (24 * r * B_beta * Z) / (gamma * torch.sqrt(torch.tensor(m))) * torch.sqrt(3 * torch.log(Q))
                rademacher_bound = first_term + second_term

                return rademacher_bound

            for epoch in tqdm(range(1, epochs + 1)):
                loss = train()
                train_acc = test(train_loader) * 100.0
                test_acc = test(test_loader) * 100.0

            rad_complexity = calculate_rademacher_complexity(model=model,
                                                             dataset=train_dataset,
                                                             hidden_dim=hd,
                                                             gnn_depth=l)

            # raw_data.append({'it': it, 'test': test_acc, 'train': train_acc, 'diff': train_acc - test_acc, 'layer': l,
            #                  'Color classes': color_counts[d][l]})
            raw_data.append({'it': it, 'test': test_acc, 'train': train_acc, 'diff': train_acc - test_acc, 'layer': l,
                             'rad_complexity': rad_complexity})

            table_data[-1].append([train_acc, test_acc, train_acc - test_acc, rad_complexity])



    data = pd.DataFrame.from_records(raw_data)
    table_data = np.array(table_data)

    with open(dataset_name + '.csv', 'w') as file:
        writer = csv.writer(file, delimiter=' ', lineterminator='\n')

        for i, h in enumerate(num_layers):
            train = table_data[i][:, 0]
            test = table_data[i][:, 1]
            diff = table_data[i][:, 2]
            rad_complexity = table_data[i][:, 3]

            writer.writerow([str(h)])
            writer.writerow(["###"])
            writer.writerow([train.mean(), train.std()])
            writer.writerow([test.mean(), test.std()])
            writer.writerow([diff.mean(), diff.std()])

            print(str(h))
            print("###")
            print(train.mean(), train.std())
            print(test.mean(), test.std())
            print(diff.mean(), diff.std())
            print(rad_complexity[-1])
