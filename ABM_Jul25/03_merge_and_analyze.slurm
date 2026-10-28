#!/bin/bash
# =============================================================================
# 03_merge_and_analyze.slurm
#
# Phase 3 of 3: run ONCE, after every array task from phase 2 has finished.
# Stitches all chunk_*.npz files into one checkpoint, computes Sobol indices
# per summary statistic, and writes heatmaps/rankings/identifiability CSVs.
# Cheap and fast — no ABM runs happen here.
#
# Submit with:  sbatch 03_merge_and_analyze.slurm
# (Or, better, use submit_all.sh which chains all three phases automatically
#  with --dependency so you don't have to babysit the queue.)
# =============================================================================

#SBATCH --job-name=merg_analze
#SBATCH --output=logs/03_analyze_%j.out
#SBATCH --error=logs/03_analyze_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=scu-cpu

set -euo pipefail
mkdir -p logs
source "/athena/cayuga_0014/scratch/kch4018/abm-files/TME_Simulation/ABM_Jul25/config.sh"

source ~/.bashrc
conda activate abm

export PYTHONPATH=/athena/cayuga_0014/scratch/kch4018/abm-files/TME_Simulation

echo "Merging chunks..."
python -u -m ABM_Jul25.sensitivity_analysis \
    --mode merge \
    --N "${N}" \
    --output-dir "${OUTPUT_DIR}" \
    --n-chunks "${N_CHUNKS}"

echo "Running Sobol analysis and generating plots..."
python -u -m ABM_Jul25.sensitivity_analysis \
    --mode analyze \
    --N "${N}" \
    --output-dir "${OUTPUT_DIR}"

end_time=$(date "+%Y-%m-%d %H:%M:%S")

echo "End time: $end_time"

echo "All done. See ${OUTPUT_DIR}/ for:"
echo "  *_sobol.csv                    - per-statistic Sobol indices"
echo "  sensitivity_heatmap_ST.pdf/png - total-order index heatmap"
echo "  sensitivity_heatmap_S1.pdf     - first-order index heatmap"
echo "  parameter_rankings.pdf         - top parameters per statistic"
echo "  identifiability_summary.csv    - FIT vs FIX recommendations"
