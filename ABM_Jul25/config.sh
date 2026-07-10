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

# Saltelli base sample size. Total parameter sets evaluated = N*(2*D+2),
# where D = number of parameters (40 in the current PARAM_SPECS in
# sensitivity_analysis.py). With D=40:
#   N=32  ->  2,624 parameter sets   (fast pilot run)
#   N=64  ->  5,248 parameter sets   (recommended first full pass)
#   N=128 -> 10,496 parameter sets   (final/high-confidence pass)
N=64

# Number of shards to split the sample matrix into for the array job.
# Must match the #SBATCH --array range in 02_evaluate_array.slurm (see note
# above). Sizing guidance (measured: ~7.5 sec/ABM-step at 'standard' scenario
# scale, so ~37.5 min per parameter set at 3 replicates x 100 steps):
#   samples_per_chunk = N*(2*D+2) / N_CHUNKS
#   wall_time_per_chunk ~= ceil(samples_per_chunk / cpus_per_task) * 37.5 min
# Pick N_CHUNKS so wall_time_per_chunk comfortably fits your partition's
# walltime limit (see --time in 02_evaluate_array.slurm).
N_CHUNKS=131

# Replicates per parameter set (handles ABM stochasticity). 3-5 is typical;
# each replicate reruns the full scenario with a different random seed.
N_REPLICATES=3

# Where all outputs (samples, chunks, checkpoint, plots, CSVs) are written.
OUTPUT_DIR=sensitivity_results

# Seed for the Saltelli sampler itself (not the ABM's internal replicate
# seeds, which are derived per-sample inside run_abm()).
SEED=42
