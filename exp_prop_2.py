from src.utils import partition_by_invariant, prop2_bounds, wl_colors
from src.datasets import sample_dataset_regular_mixed
from src.net import GIN
import torch
import random
import numpy as np
import argparse
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-seed", type=int, required=True)
    p.add_argument("--mc-seed", type=int, required=True)
    p.add_argument("--m", type=int, default=100)
    p.add_argument("--n", type=int, default=12, help="node count for the d-regular graphs")
    p.add_argument("--d", type=int, default=3, help="regular degree")
    p.add_argument("--q", type=int, required=True, help="number of WL graphs out of m")
    p.add_argument("--prop", type=float, default=None,
                    help="proportion of m per rich class (default: 1 copy per rich class, old behavior)")
    p.add_argument("--hidden-channels", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--wl-iterations", type=int, default=2, help="should be >= num_layers (see Xu et al.)")
    p.add_argument("--activation", type=str, default="relu", help="Choose relu or leaky_relu")
    p.add_argument("--dropout", type=float, default=0.0, help="dropout prob inside each GIN MLP (default 0: off)")
    p.add_argument("--K", type=int, default=20, help="number of Rademacher sigma draws")
    p.add_argument("--epochs", type=int, default=150, help="Adam steps per (draw, restart)")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true", help="also log this run to Weights & Biases")
    p.add_argument("--wandb-project", type=str, default="rad-prop2")
    p.add_argument("--wandb-entity", type=str, default=None)
    return p.parse_args()


def estimate_rademacher(dataset, K: int, hidden_channels:int, num_layers:int, activation: str = 'relu', restarts: int = 2, epochs: int = 150, lr: float = 0.05, device: str = "cpu", seed: int = 42, grad_clip: bool = False, dropout: float = 0.0):
    torch.manual_seed(seed)
    random.seed(seed)
    m = len(dataset)
    values = []
    eps=1e-7
    # Do K Rademacher draws
    for k in tqdm(range(K)):
        sigma = torch.tensor(
            [1.0 if random.random() < 0.5 else -1.0 for _ in range(m)], device=device,
        )
        y01 = (sigma + 1.0) / 2.0
        best = -1e9
        # Number of initialization tries
        for _ in range(restarts):
            model = GIN(
                    hidden_channels=hidden_channels, num_layers=num_layers,
                    activation=activation, dropout=dropout
            ).to(device=device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            best_per_k = -1e9
            for _ in range(epochs):
                model.train()
                opt.zero_grad()
                preds = model.forward_batch(dataset)
                # # Our loss is basically the Rad
                # # 1/m \sum \sigma * f()
                # obj = (sigma * perds).mean()
                # # We do - to get the sup
                # (-obj).backward()
                #p = ((preds + 1.0) / 2.0).clamp(eps, 1.0 - eps)
                preds = torch.nn.functional.sigmoid(preds)
                #print(preds)
                loss = torch.nn.functional.binary_cross_entropy(preds, y01)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                model.eval()
                with torch.no_grad():
                    y_hat = torch.nn.functional.sigmoid(model.forward_batch(dataset)) > 0.5
                    acc = (y01.bool() == y_hat).sum().float().item() / len(dataset)
                    obj = acc
                    # obj = (sigma * model.forward_batch(dataset)).mean().item()
                best = max(best, obj)
                best_per_k = max(best_per_k, obj)
            #print(f"k:{k} -> Acc: {best_per_k}")

        values.append(best) 

    mc_estimate = (2 * float(np.mean(values))) - 1
    return mc_estimate, values

def main():
    args = parse_args()

    data_seed = args.data_seed
    mc_seed = args.mc_seed
    m = args.m # Number of samples
    n = args.n # Number of vertices
    d = args.d # Degree of regularity
    q = args.q # Number of additional 1WL graphs
    K = args.K # Number of repeats for MC

    device = args.device

    graphs = sample_dataset_regular_mixed(m, n, d, q, data_seed, prop=args.prop)
    invariants = wl_colors(graphs, iterations=args.wl_iterations)
    groups = partition_by_invariant(invariants)

    p = len(groups) # Number of partitions
    mu_counts = [len(v) for v in groups.values()] # len(p) of each partition
    mu_list = [mu / args.m for mu in mu_counts]

    lower_rad, upper_rad, exact_rad = prop2_bounds(mu_list,mu_counts, m)
    torch_graphs = [torch.tensor(A, dtype=torch.float32, device=args.device) for A in graphs]

    mc_rad, vals = estimate_rademacher(
        dataset=torch_graphs, hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        activation=args.activation,
        K=K, restarts=args.restarts, epochs=args.epochs,
        lr=args.lr, device=device, seed=mc_seed, dropout=args.dropout
    )

    mc_std = float(torch.tensor(vals).std().item()) if len(vals) > 1 else 0.0
    in_band = lower_rad <= mc_rad <= upper_rad

    row = dict(
        p=p, q=q, prop=args.prop, n=n, d=d, m=m, K=args.K,
        dataset_seed=args.data_seed, mc_seed=args.mc_seed,
        hidden_dim=args.hidden_channels, num_layers=args.num_layers, wl_iterations=args.wl_iterations, dropout=args.dropout,
        epochs=args.epochs, lr=args.lr, restarts=args.restarts,
        lower_rad=lower_rad, upper_rad=upper_rad, exact_rad=exact_rad, mc_estimate=mc_rad, mc_std=mc_std,
        in_band=in_band, device=device,
    )
    print(row)

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        wandb.log(row)
        wandb.finish()

if __name__ == "__main__":
    main()
