# Rademacher Meets Colors

This repository contains the source code for experiments accompanying the research presented in [**"Rademacher Meets Colors: More Expressivity, but at What Cost?"** (arXiv:2510.10101)](https://arxiv.org/abs/2510.10101).

In these experiments, we empirically evaluate the generalization bounds and generalization gap of GNN model performance across standard datasets such as TUDatasets. We investigate how moving up this hierarchy (from 1-WL to higher-order variants) impacts the generalization bounds discussed in the paper.


## Quick Start

To set up the environment and run the experiments, you can use the provided configuration files:

```bash
# Using Conda
conda env create -f environment.yml
conda activate wl_meets_rc

# Using Pip
pip install -r requirements.txt

```


## Basic Usage

The main entry point for the experiments is `run_experiment.py`. This script handles data fetching via PyTorch Geometric's `TUDataset` interface, model initialization, and the training/evaluation loop.

To run the experiment:

```bash
python run_experiment.py

```

### A Note on TUDatasets

The script automatically downloads the requested dataset into a `/data` directory upon the first execution. When you run the script, `torch_geometric.datasets.TUDataset` will check for the raw files locally. If they aren't found, it will pull them from the [TU Dortmund University](https://chrsmrrs.github.io/datasets/) servers. Ensure you have an active internet connection for the initial run. 
