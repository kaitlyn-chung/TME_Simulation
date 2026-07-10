#!/bin/bash
# =============================================================================
# 02_evaluate_array.slurm
#
# Phase 2 of 3: the expensive part. Each array task independently evaluates
# ONE shard ("chunk") of the sample matrix and writes its own chunk file, so
# tasks never step on each other and a failed/killed task only costs you that
# one chunk (re-run just that --array index to redo it).
#
# --- Sizing this job: the numbers that matter -------------------------------
# Measured on this ABM (30x30 grid, 'standard' scenario): ONE model.step()
# takes ~7.5 sec, dominated by the diffusion solver's fixed inner loop
# (21,600 micro-steps x 9 cytokine fields per macro-step) — this cost is
# roughly independent of grid size. So:
#     100 steps/replicate  ->  ~12.5 min/replicate
#     3 replicates/param-set -> ~37.5 min/parameter set (run serially inside
#                                one process — see run_abm() in the script)
#
# A single core --cpus-per-task=N_CORES processes N_CORES parameter sets
# concurrently (via Python multiprocessing.Pool inside evaluate-chunk).
# Pick CHUNK_SIZE (param sets per array task) and --time so that:
#     --time  >~  ceil(CHUNK_SIZE / CPUS_PER_TASK) * 37.5 min  (+ margin)
#
# Example: N=64 -> 5,248 parameter sets total.
#   N_CHUNKS=131, CPUS_PER_TASK=8  -> ~40 sets/chunk -> 5 waves x 37.5min ~ 3.1h
#   Set --time=04:00:00 for margin, and tune N_CHUNKS to your partition's
#   walltime cap and how many array tasks can run concurrently.
#
# Submit with:  sbatch 02_evaluate_array.slurm
# (Run 01_generate_samples.slurm first — this job reads its output.)
# =============================================================================

#SBATCH --job-name=eval
#SBATCH --output=logs/02_eval_%A_%a.out
#SBATCH --error=logs/02_eval_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=scu-cpu
#
# IMPORTANT: #SBATCH directives are parsed by Slurm BEFORE the script runs,
# so this array range CANNOT read N_CHUNKS from config.sh — it must be kept
# in sync with it BY HAND. If you change N_CHUNKS in config.sh, update the
# range below to match (0 to N_CHUNKS-1). They currently assume N_CHUNKS=131.
#
# The "%20" throttle limits how many array tasks run at once — raise it if
# your allocation/QOS allows more concurrency, lower it if you get throttled
# or your admins ask you to play nice with shared resources.
#SBATCH --array=0-130%20

set -euo pipefail
mkdir -p logs
source "/athena/cayuga_0014/scratch/kch4018/abm-files/TME_Simulation/ABM_Jul25/config.sh"

source ~/.bashrc
conda activate abm

CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}

echo "Evaluating chunk ${SLURM_ARRAY_TASK_ID} of ${N_CHUNKS} using ${CPUS_PER_TASK} cores..."

export PYTHONPATH=/athena/cayuga_0014/scratch/kch4018/abm-files/TME_Simulation

python -u -m ABM_Jul25.sensitivity_analysis \
    --mode evaluate-chunk \
    --N "${N}" \
    --output-dir "${OUTPUT_DIR}" \
    --n-chunks "${N_CHUNKS}" \
    --chunk-index "${SLURM_ARRAY_TASK_ID}" \
    --n-jobs "${CPUS_PER_TASK}" \
    --n-replicates "${N_REPLICATES}"

echo "Chunk ${SLURM_ARRAY_TASK_ID} done."
