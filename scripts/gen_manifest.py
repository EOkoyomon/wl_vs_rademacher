M_VALUES = [50, 100, 200] # Numbre of samples
P_VALUES = [2, 3, 4, 5] # Number of WL classes (p=1 excluded -- no rich classes to speak of)
N_LAYERS = [1]
HIDDEN_CHANNELS = [8, 16, 32]
PROP_VALUES = [0.01, 0.05, 0.1, 0.2]  # proportion of m per q class
SEEDS = 1
FIXED = "--n 12 --d 3 --wl-iterations 3 --K 100 --epochs 50 --lr 0.05 --restarts 10 --device cuda --activation leaky_relu --wandb"

lines = []
for m in M_VALUES:
    for p in P_VALUES:
        for hidden_channels in HIDDEN_CHANNELS:
            q = p - 1
            if q > m:
                continue
            for prop in PROP_VALUES:
                if prop * q >= 1.0:
                    continue
                for rep in range(SEEDS):
                    data_seed, mc_seed = 1000 + rep, 2000 + rep
                    lines.append(
                        f"uv run --extra cu130 exp_prop_2.py --hidden-channels {hidden_channels} --data-seed {data_seed} --mc-seed {mc_seed} "
                        f"--m {m} --q {q} --prop {prop} {FIXED}"
                    )

with open("jobs.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {len(lines)} jobs to jobs.txt")
print(f"Run: mkdir -p results logs && sbatch --array=1-{len(lines)} slurm_prop2_sweep.sbatch")
