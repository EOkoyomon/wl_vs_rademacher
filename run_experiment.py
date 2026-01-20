import csv
import os.path as osp
import random
import wandb
import argparse

import numpy as np
from torch.nn import BCEWithLogitsLoss
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm
from src.utils import OneHotNodeLabel, CalculateWLColors
from src.net import Net, GCN
from collections import defaultdict

batch_size = 128
num_layers = [0, 1, 2, 3, 4, 5, 6]
lr = 0.001
epochs = 500
lipschits_constant_loss = 1.0
delta_prob = 0.95
dataset_name_list = {
    # "MCF-7": [11533, 25417, 26872, 27048, 27059, 0, 0],
    "NCI1": [2889, 3906, 4027, 4039, 4039, 4039, 4039],
    # "MUTAGENICITY": [2819, 3624, 4239, 4317, 4317, 4317, 4317],
}
#seeds = [1, 2, 3, 4, 5] 
num_reps = 5
hd = 64
early_stopping = 10

sample_sizes = [50, 100]


def main(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print("Using device", device)



    for d, dataset_name in enumerate(dataset_name_list.keys()):
        print("Loading dataset", dataset_name, end='... ')
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'TU')
        transform = Compose([CalculateWLColors(num_layers)])
        dataset = TUDataset(path, name=dataset_name, pre_transform=transform, force_reload=True).shuffle()
        print("done")

        raw_data = []
        table_data = []

        diffs = []
        diffs_std = []

        for l in num_layers:
            print("Number of layers:", l)
            table_data.append([])
            for m in sample_sizes + [len(dataset) - (len(dataset) // 10)]:
                print("Number of samples:", m)
                config = {
                    "delta_prob": delta_prob,
                    "m": m,
                    "lr": lr,
                    "num_layers": num_layers,
                    "epochs": epochs,
                    "hidden_dimension": hd,
                    "early_stopping": early_stopping,
                    "batch_size": batch_size
                }
                with wandb.init(name="GCN", project="wl_meet_rad", entity="ai-re", config=config) as run:
                    start_train = len(dataset) // 10
                    dataset.shuffle()
                    p_dict = defaultdict(int)

                    for g in dataset[start_train:start_train+m]:
                        p_dict[g[f'wl_hash_{l}']] += 1

                    p_theory_upper_bound = 0.0
                    p_theory_lower_bound = 0.0

                    for c_j in p_dict.keys():
                        mu_j = p_dict[c_j]
                        p_theory_upper_bound += np.sqrt(mu_j)
                        p_theory_lower_bound += np.sqrt(mu_j / 2.0)

                    gen_err_upper_bound = 2.0 * lipschits_constant_loss * p_theory_upper_bound * (1.0/m) + 3.0 * np.sqrt(np.log(2.0/delta_prob)/2.0*m)
                    gen_err_lower_bound = 2.0 * lipschits_constant_loss * p_theory_lower_bound * (1.0/m) + 3.0 * np.sqrt(np.log(2.0/delta_prob)/2.0*m)



                    train_dataset = dataset[start_train:start_train+m]
                    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

                    test_dataset = dataset[:len(dataset) // 10]
                    test_loader = DataLoader(test_dataset, batch_size)

                    # model = Net(dataset.num_features, hd, 1, l).to(device) # Binary
                    model = GCN(dataset.num_features, hd, 1, l).to(device) # Binary
                    print("Model params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    def train():
                        loss_fn = torch.nn.BCEWithLogitsLoss()
                        model.train()

                        total_loss = 0
                        for data in train_loader:
                            data = data.to(device)
                            optimizer.zero_grad()
                            out = model(data.x, data.edge_index, data.batch)
                            # print(out)
                            labels = data.y.to(torch.float32).view(out.shape) # Convert data.y to float from int
                            loss = loss_fn(out, labels) #population_risk(out, labels)
                            loss.backward()
                            optimizer.step()
                            total_loss += float(loss) * data.num_graphs
                        return total_loss / len(train_loader.dataset)


                    @torch.no_grad()
                    def test(loader):
                        model.eval()
                        loss_fn = torch.nn.BCEWithLogitsLoss()

                        total_correct = 0
                        total_loss = 0
                        for data in loader:
                            data = data.to(device)
                            pred = model(data.x, data.edge_index, data.batch)
                            labels = data.y.to(torch.float32).view(pred.shape)
                            loss = loss_fn(pred, labels)
                            # pred = torch.argmax(pred, dim=1)
                            pred = (pred.squeeze(-1) > 0.5).long()
                            total_correct += int((pred == data.y).sum())
                            total_loss += float(loss) * data.num_graphs
                        return total_correct / len(loader.dataset), total_loss / len(loader.dataset)
                    
                    last_best_train = 1000
                    epoch = 0
                    last_update_train = 0

                    while True:
                        loss = train()
                        train_acc, train_loss = test(train_loader)
                        test_acc, test_loss = test(test_loader)

                        train_acc *= 100.0
                        test_acc *= 100.0

                        is_early_stop = last_update_train == early_stopping
                        if 100 - train_acc < 1e-2 or is_early_stop or epoch == epochs:
                            print(f"Stopped at acc {train_acc} and ES {is_early_stop}")
                            break

                        if loss < last_best_train:
                            last_best_train = loss
                            last_update_train = 0

                        epoch += 1
                        last_update_train += 1
                        run.log({
                            "epoch": epoch,
                            "train_acc": train_acc,
                            "test_acc": test_acc,
                            "train_loss": train_loss,
                            "test_loss": test_loss,
                            "gen_error_loss": train_loss - test_loss,
                            "gen_err_acc": train_acc - test_acc,

                        })

                    run.log({
                        "R_s_upper_bound": p_theory_upper_bound,
                        "R_s_lower_bound": p_theory_lower_bound,
                        "gen_error_upper_bound": gen_err_upper_bound,
                        "gen_error_lower_bound": gen_err_lower_bound,
                    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Argument parser for experiment configurations."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for RNG",
    )
    args = parser.parse_args()
    main(args.seed)
