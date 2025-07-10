import csv
import os.path as osp

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, global_add_pool, GraphConv
from tqdm import tqdm

batch_size = 128
# num_layers = [0, 1, 2, 3, 4, 5, 6]
num_layers = [0]
lr = 0.001
epochs = 500
dataset_name_list = ["ENZYMES"]
num_reps = 3
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
            self.mlp = MLP([hidden_channels, hidden_channels, out_channels])
        else:
            self.mlp = MLP([in_channels, hidden_channels, out_channels])

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
        x = global_add_pool(x, batch)

        return torch.sigmoid(self.mlp(x)) #return self.mlp(x) was before


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

            model = Net(dataset.num_features, hd, dataset.num_classes, l).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)


            def train():
                model.train()

                total_loss = 0
                for data in train_loader:
                    data = data.to(device)
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index, data.batch)
                    loss = F.cross_entropy(out, data.y)
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
                    pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
                    total_correct += int((pred == data.y).sum())
                return total_correct / len(loader.dataset)
            
            def max_l2_norms_weights(model):
                w1_norms = []
                w2_norms = []
                
                for conv in model.convs:
                    # Assuming conv.lin and conv.root are the weight layers
                    # conv.lin is used for neighbors (W2), conv.root is used for the root/self node (W1)
                    W2 = conv.lin.weight
                    W1 = conv.root.weight
                    
                    w2_norm = torch.norm(W2, p=2)
                    w1_norm = torch.norm(W1, p=2)
                    
                    w2_norms.append(w2_norm.item())
                    w1_norms.append(w1_norm.item())
                
                return max(w1_norms), max(w2_norms)
            
            def calculate_rademacher_complexity(model, train_loader, m):
                # r is the dimension of the embedding
                r = hd

                # d is the branching factor (i.e. max number of neighbours for any node)
                d = None # TODO: Calculate using train_dataset.

                # m is the sample size
                m = train_dataset.len()

                # L is the depth of the GNN
                L = l

                # C_rho is the Lipschitz constant of the transform (rho) of the locally-invariant neighbourhood aggregations. The paper does not indicate nonlinear but the survey paper says nonlinear.
                C_rho = 1 # ReLU, but using identity in the code by default
                
                # C_g is the Lipschitz constant of the nonlinear transform (g) applied to each neighbour before aggregation
                C_g = 1 # ReLU, but using identity in the code by default
                
                # C_phi is the Lipschitz constant of the non-linear transform (phi) applied at the end of the embedding update
                C_phi = 1 # Tanh

                # C_B1 is the upper bound of the L2 norm of the weight matrix that is applied to the main node embedding (x_v) before combination with its neighbours. The survey paper applies it to h_v rather than x_v and bounds this.
                # C_B2 is the upper bound of the L2 norm of the weight matrix that is applied to the aggregated neighbours, after the rho transformation.
                C_B1, C_B2 = max_l2_norms_weights(model)

                # B_x is the upper bound of the L2 norm of the feature vector x_v
                B_x = 1 # TODO: Add correct bound.

                # B_beta is the upper bound of the L2 norm of weight matrix, Beta, of the readout function, used to apply a binary classifier by converting from feature dim to 1 dim.
                B_beta = torch.norm(model.mlp, p=2)
                
                # TODO: Calculate the bound
                pass


            for epoch in tqdm(range(1, epochs + 1)):
                loss = train()
                train_acc = test(train_loader) * 100.0
                test_acc = test(test_loader) * 100.0

            # raw_data.append({'it': it, 'test': test_acc, 'train': train_acc, 'diff': train_acc - test_acc, 'layer': l,
            #                  'Color classes': color_counts[d][l]})
            raw_data.append({'it': it, 'test': test_acc, 'train': train_acc, 'diff': train_acc - test_acc, 'layer': l})

            table_data[-1].append([train_acc, test_acc, train_acc - test_acc])



    data = pd.DataFrame.from_records(raw_data)
    table_data = np.array(table_data)

    with open(dataset_name + '.csv', 'w') as file:
        writer = csv.writer(file, delimiter=' ', lineterminator='\n')

        for i, h in enumerate(num_layers):
            train = table_data[i][:, 0]
            test = table_data[i][:, 1]
            diff = table_data[i][:, 2]
            # color = table_data[i][:, 3]

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
            # print(color[-1])
