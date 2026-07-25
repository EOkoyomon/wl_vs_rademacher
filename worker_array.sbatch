#!/bin/bash
#SBATCH --job-name=rad-prop2
#SBATCH --output=logs/RAD/%A_%a.out
#SBATCH --error=logs/RAD/%A_%a.err
#SBATCH --nodelist=diufrd205
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=GPU

source  ./.venv/bin/activate

set -euo pipefail
cmd=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs.txt)
echo "Running: $cmd"
eval "$cmd"
