#!/bin/bash
# =============================================================================
# 01_generate_samples.slurm
#
# Phase 1 of 3: generate the Saltelli sample matrix ONCE and save it to disk,
# so every array task in phase 2 and the merge/analyze step in phase 3 all
# read the exact same parameter sets. This job is cheap (<1 min) — it does
# NOT run the ABM, just SALib's sampler.
#
# Submit with:  sbatch 01_generate_samples.slurm
# =============================================================================

#SBATCH --job-name=sampler
#SBATCH --output=logs/01_sample_%j.out
#SBATCH --error=logs/01_sample_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=scu-cpu

set -euo pipefail
mkdir -p logs
source "/athena/cayuga_0014/scratch/kch4018/abm-files/TME_Simulation/ABM_Jul25/config.sh"

source ~/.bashrc
conda activate abm

export PYTHONPATH=/athena/cayuga_0014/scratch/kch4018/abm-files/TME_Simulation

python -u -m ABM_Jul25.sensitivity_analysis \
    --mode sample \
    --N "${N}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}"

echo "Sample generation complete. Check ${OUTPUT_DIR}/param_values_N${N}.npy"
echo "Next: submit 02_evaluate_array.slurm with --n-chunks sized per the printed suggestion above."
