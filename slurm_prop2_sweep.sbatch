#!/bin/bash
# Generate jobs.txt first:  python gen_manifest.py
# Submit:                   sbatch --array=1-$(wc -l < jobs.txt) slurm_prop2_sweep.sbatch
#SBATCH --job-name=rad-prop2-sweep
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
# #SBATCH --partition=<your-partition>
# #SBATCH --account=<your-account>

set -euo pipefail
mkdir -p logs results

cmd=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs.txt)
echo "Running: $cmd"
eval "$cmd"
