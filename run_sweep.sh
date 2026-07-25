#!/bin/bash
# The one command to run: ./run_sweep.sh
# Generates jobs.txt, then submits it as a GPU SLURM array.
set -euo pipefail

source  ./.venv/bin/activate

mkdir -p results logs
python gen_manifest.py

N=$(wc -l < jobs.txt)
sbatch --array=1-${N} worker_array.sbatch
