M_VALUES = [50, 100, 200] # Numbre of samples
P_VALUES = [1, 2, 3, 4, 5] # Number of WL classes
N_LAYERS = [3, 4, 5, 6]
SEEDS = 1
FIXED = "--n 12 --d 3 --wl-iterations 3 --K 100 --epochs 30 --lr 0.05 --restarts 10 --device cuda --activation leaky_relu --wandb"

lines = []
for m in M_VALUES:
    for p in P_VALUES:
        for n_layers in N_LAYERS:
            q = p - 1
            if q > m:
                continue
            for rep in range(SEEDS):
                data_seed, mc_seed = 1000 + rep, 2000 + rep
                lines.append(
                    f"uv run --extra cu130 exp_prop_2.py --num--layers {n_layers} --data-seed {data_seed} --mc-seed {mc_seed} "
                    f"--m {m} --q {q} {FIXED}"
                )

with open("jobs.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {len(lines)} jobs to jobs.txt")
print(f"Run: mkdir -p results logs && sbatch --array=1-{len(lines)} slurm_prop2_sweep.sbatch")
