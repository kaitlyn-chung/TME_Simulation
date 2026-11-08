#!/bin/bash
# =============================================================================
# config.sh — single source of truth for run parameters shared by
# 01_generate_samples.slurm, 02_evaluate_array.slurm, and
# 03_merge_and_analyze.slurm. Edit values here rather than in each script.
#
# ONE EXCEPTION: the #SBATCH --array=... line in 02_evaluate_array.slurm is
# a Slurm directive parsed before any shell code runs, so it CANNOT read
# N_CHUNKS from this file. If you change N_CHUNKS below, you must also
# update the --array=0-<N_CHUNKS-1> line at the top of 02_evaluate_array.slurm
# by hand (there's a reminder comment there too).
# =============================================================================

# Saltelli base sample size. Total parameter sets evaluated = N*(2*D+2), if including second order
# where D = number of parameters (48 in the current PARAM_SPECS in
# sensitivity_analysis.py). If no second order calculations N*(D+2) parameter sets are evlauted.
N=128

# Number of shards to split the sample matrix into for the array job.
# Must match the #SBATCH --array range in 02_evaluate_array.slurm.
# Pick N_CHUNKS so wall_time_per_chunk comfortably fits your partition's
# walltime limit (see --time in 02_evaluate_array.slurm).
N_CHUNKS=100

# Replicates per parameter set (handles ABM stochasticity). 3-5 is typical;
# each replicate reruns the full scenario with a different random seed.
N_REPLICATES=5

# Where all outputs (samples, chunks, checkpoint, plots, CSVs) are written.
OUTPUT_DIR=/athena/cayuga_0014/scratch/kch4018/abm-files/TME_Simulation/ABM_Jul25/sensitivity_results

# Seed for the Saltelli sampler itself (not the ABM's internal replicate
# seeds, which are derived per-sample inside run_abm()).
SEED=42
