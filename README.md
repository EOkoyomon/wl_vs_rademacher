# Rademacher Meets Colors

This repository contains the source code for experiments accompanying the research presented in [**"Rademacher Meets Colors: More Expressivity, but at What Cost?"** (arXiv:2510.10101)](https://arxiv.org/abs/2510.10101).

In these experiments, we empirically evaluate the generalization bounds and generalization gap of GNN model performance across standard datasets such as TUDatasets. We investigate how moving up this hierarchy (from 1-WL to higher-order variants) impacts the generalization bounds discussed in the paper.


## Quick Start

To set up the environment and run the experiments:

```bash
# Using uv

# CPU
uv sync --extra cpu
# GPU
uv sync --extra cu130

```

## Basic Usage

We test the following theoretical results
## Prop 2 - Upper and Lower bounds

The file containing the experiment is `exp_prop_2.py`. This experiment trains a GIN network with `activation` 
activations on a dataset of `m` samples with `m-q` of `d`-regular graphs and
`q` 1-WL distinguishable graphs on `num-layers` where the WL distinguishability is given on `wl-iterations`. 


Then, to estimate #\mathcal{R}_S(\mathcal{F}_A)$ we use Monte Carlo sampling and train the network `K` times
on random assignment of Rademaher variables `\sigma` to the datapoints with a BCE loss and then average 
over the `K` runs. To attain the $\mathrm{sup}$ in the expectation we perform `restarts` restarts to get
different parameter initializations.

The results show the theoretical bands for $\mathcal{R}_S$ along the closed-form solution.

```bash
uv run --extra cpu exp_prop_2.py --data-seed 1000 --mc-seed 2000 --m 100 --n 12 --d 3 --q 1 --num-layers 3 --wl-iterations 3 --K 80 --epochs 30 --lr 0.05 --restarts 2 --device cpu --activation leaky_relu
```
