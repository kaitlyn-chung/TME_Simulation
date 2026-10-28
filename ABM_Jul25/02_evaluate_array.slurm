#!/bin/bash
# =============================================================================
# 02_evaluate_array.slurm
#
# Phase 2 of 3: the expensive part. Each array task independently evaluates
# ONE shard ("chunk") of the sample matrix and writes its own chunk file, so
# tasks never step on each other and a failed/killed task only costs you that
# one chunk (re-run just that --array index to redo it).
# In the past, running 400 chunks for 3200 parameter sets has a maximum walltime of 3.5 hours
# Submit with:  sbatch 02_evaluate_array.slurm
# (Run 01_generate_samples.slurm first — this job reads its output.)
# =============================================================================

#SBATCH --job-name=eval
#SBATCH --output=logs/02_eval_%A_%a.out
#SBATCH --error=logs/02_eval_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=scu-cpu
#
# IMPORTANT: #SBATCH directives are parsed by Slurm BEFORE the script runs,
# so this array range CANNOT read N_CHUNKS from config.sh — it must be kept
# in sync with it BY HAND. If you change N_CHUNKS in config.sh, update the
# range below to match (0 to N_CHUNKS-1). 
#
# The "%37" throttle limits how many array tasks run at once. On the cluster
# the maximum CPU useage per user is 300 CPUs, so 37 x 8 = 296 CPUs used at once
#SBATCH --array=0-799%37

set -euo pipefail
mkdir -p logs
source "/athena/cayuga_0014/scratch/kch4018/abm-files/TME_Simulation/ABM_Jul25/config.sh"

source ~/.bashrc
conda activate abm

CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}

echo "Evaluating chunk ${SLURM_ARRAY_TASK_ID} of ${N_CHUNKS} using ${CPUS_PER_TASK} cores..."

start_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_time"

export PYTHONPATH=/athena/cayuga_0014/scratch/kch4018/abm-files/TME_Simulation

python -u -m ABM_Jul25.sensitivity_analysis \
    --mode evaluate-chunk \
    --N "${N}" \
    --output-dir "${OUTPUT_DIR}" \
    --n-chunks "${N_CHUNKS}" \
    --chunk-index "${SLURM_ARRAY_TASK_ID}" \
    --n-jobs "${CPUS_PER_TASK}" \
    --n-replicates "${N_REPLICATES}"

end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "End time: $end_time"
echo "Chunk ${SLURM_ARRAY_TASK_ID} done."
