#!/bin/bash
# The labelled beamlet pool on DelftBlue: one array task = one shard of the plan.
#   sbatch --array=0-9 scripts/slurm/beamlet_pool_delftblue.sbatch
# Each task takes the batches b with b % SLURM_ARRAY_TASK_COUNT == SLURM_ARRAY_TASK_ID,
# skips finished ones and resumes an interrupted one beamlet by beamlet, so the same
# array can simply be resubmitted after a time limit. Set the account and the paths
# for your project; scripts/docs/beamlet_pool.md lists what must exist beforehand.
#SBATCH --job-name=adota-beamlet-pool
#SBATCH --partition=compute
#SBATCH --account=<education-or-research-account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3G
#SBATCH --time=120:00:00
#SBATCH --output=logs/beamlet_pool_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export TMPDIR="${TMPDIR:-/scratch/${USER}/tmp}"
mkdir -p "${TMPDIR}" logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

uv run python scripts/al_build_beamlet_pool.py generate \
    --config scripts/config_beamlet_pool_delftblue.yaml \
    --shard slurm --num-threads "${SLURM_CPUS_PER_TASK}"
