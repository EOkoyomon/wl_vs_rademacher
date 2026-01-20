import torch
from torch_geometric.nn import Linear, global_mean_pool, GraphConv, GCNConv, MLP, global_add_pool
from torch_geometric.utils import scatter

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nc):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(nc):
            # self.convs.append(
            #     GCNConv(in_channels, hidden_channels, aggr="add", bias=True)
            # )
            self.convs.append(GraphConv(in_channels, hidden_channels, aggr='add', bias=True))
            in_channels = hidden_channels

        # if nc != 0:
        #     self.readout = Linear(hidden_channels, out_channels)
        # else:
        #     self.readout = Linear(in_channels, out_channels)

        if nc != 0:
            self.readout = MLP([hidden_channels, hidden_channels, out_channels])
        else:
            self.readout = MLP([in_channels, hidden_channels, out_channels])

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))

        x = global_add_pool(x, batch)
        
        return self.readout(x)

class GCNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm, residual):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual

        self.batchnorm_h = torch.nn.BatchNorm1d(out_dim)
        self.conv = GCNConv(in_dim, out_dim, add_self_loops=False, normalize=True)
    def forward(self, x, edge_index):
        h_in = x

        h = self.conv(x=x, edge_index=edge_index)

        if self.batch_norm:
            h = self.batchnorm_h(h)

        h = torch.nn.functional.relu(h)

        if self.residual:
            h = h_in + h

        return h

class MLPReadout(torch.nn.Module):
    def __init__(self, input_dim, output_dim, L=2):
        super().__init__()
        list_FC_layers = [torch.nn.Linear(input_dim//2**l, input_dim//2**(l+1), bias=True) for l in range(L)]
        list_FC_layers.append(torch.nn.Linear(input_dim//2**L, output_dim, bias=True))
        self.FC_layers = torch.nn.ModuleList(list_FC_layers)

        self.L = L
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = torch.nn.functional.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers):
        super().__init__()
        self.encoder = MLP(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=hidden_channels, num_layers=2)

        self.layers = torch.nn.ModuleList([
            GCNLayer(in_channels, hidden_channels, batch_norm=True, residual=True)
            for _ in range(n_layers-1)
            ]
        )
        self.readout = MLPReadout(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        h = self.encoder(x)

        for layer in self.layers:
            h = layer(h, edge_index)

        h = scatter(h, batch, reduce="sum")

        h = self.readout(h)

        return h
            


