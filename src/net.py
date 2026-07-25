import torch
from torch_geometric.nn import Linear, global_mean_pool, GraphConv, GCNConv, MLP, global_add_pool, GINConv
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
        self.conv = GCNConv(in_dim, out_dim, add_self_loops=True, normalize=True)
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
            GCNLayer(hidden_channels, hidden_channels, batch_norm=True, residual=True)
            for _ in range(n_layers)
            ]
        )
        self.readout = MLPReadout(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        h = self.encoder(x)

        for layer in self.layers:
            h = layer(h, edge_index)

        h = global_add_pool(h, batch)

        h = self.readout(h)

        return h
            


class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, mlp_hidden=None, activation="relu"):
        super().__init__()
        mlp_hidden = mlp_hidden or hidden_channels
        self.activation = activation
        act_layer = torch.nn.LeakyReLU(0.1) if activation =="leaky_relu" else torch.nn.ReLU()
        self.eps = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])
        self.convs = torch.nn.ModuleList()
        
        for i in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(1 if i == 0 else hidden_channels, hidden_channels),
                act_layer,
                torch.nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(mlp)
        self.beta = torch.nn.Linear(hidden_channels, 1, bias=False)

    def _act(self, x):
        if self.activation == "leaky_relu":
            return torch.nn.functional.leaky_relu(x, 0.1)
        else:
            return torch.relu(x)

    def embedding(self, A):
        n = A.shape[0]
        h = torch.ones(n, 1, dtype=A.dtype, device=A.device)
        for eps, mlp in zip(self.eps, self.convs):
            agg = A @ h
            combined = (1.0 + eps) * h + agg
            h = self._act(mlp(combined))
        return h.sum(dim=0)

    def logits(self, A):
        return self.beta(self.embedding(A)).squeeze()

    def forward(self, A):
        return torch.tanh(self.logits(A))

    def embedding_batch(self, As):
        """Same computation as embedding(), but for a whole batch of
        same-size graphs at once via torch.bmm -- one vectorized GPU call
        per layer instead of a Python loop over individual graphs (which
        is dominated by per-call kernel-launch overhead for tiny graphs)."""
        A = torch.stack(As) if isinstance(As, (list, tuple)) else As  # (B, n, n)
        B, n, _ = A.shape
        h = torch.ones(B, n, 1, dtype=A.dtype, device=A.device)
        for eps, mlp in zip(self.eps, self.convs):
            agg = torch.bmm(A, h)
            combined = (1.0 + eps) * h + agg
            h = self._act(mlp(combined))
        return h.sum(dim=1)

    def logits_batch(self, As):
        return self.beta(self.embedding_batch(As)).squeeze(-1)

    def forward_batch(self, As):
        return torch.tanh(self.logits_batch(As))


