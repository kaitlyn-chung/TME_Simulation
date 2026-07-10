#!/bin/bash
# =============================================================================
# submit_all.sh
#
# Submits all three phases of the sensitivity analysis with correct SLURM
# --dependency chaining, so you can fire-and-forget instead of babysitting
# the queue for each phase:
#
#   01_generate_samples   (1 job)
#         |  afterok
#         v
#   02_evaluate_array     (array job, many concurrent tasks)
#         |  afterok  (waits for the WHOLE array, not just task 0)
#         v
#   03_merge_and_analyze  (1 job)
#
# Usage:  ./submit_all.sh
# =============================================================================
set -euo pipefail
mkdir -p logs

cd /athena/cayuga_0014/scratch/kch4018/abm-files/TME_Simulation/ABM_Jul25

start_time=$(date +%Y-%m-%d %T)
echo "Start time: $start_time"

echo "Submitting phase 1: generate samples..."
JOB1=$(sbatch --parsable 01_generate_samples.slurm)
echo "  -> job ${JOB1}"

echo "Submitting phase 2: evaluate array (depends on job ${JOB1})..."
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} 02_evaluate_array.slurm)
echo "  -> job ${JOB2}"

echo "Submitting phase 3: merge + analyze (depends on ALL of array job ${JOB2})..."
JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} 03_merge_and_analyze.slurm)
echo "  -> job ${JOB3}"

echo ""
echo "Submitted chain: ${JOB1} -> ${JOB2} (array) -> ${JOB3}"
echo "Monitor with:  squeue -u \$USER"
echo "Logs land in:  logs/"
echo ""
echo "NOTE: if job ${JOB1} or any tasks in array job ${JOB2} fail (e.g. hit the"
echo "walltime limit), phase 3 will NOT run automatically (afterok requires"
echo "success). To resume after fixing/resubmitting failed pieces:"
echo "  - Re-run only the failed array indices:"
echo "      sbatch --array=<failed_indices> 02_evaluate_array.slurm"
echo "    (evaluate-chunk skips already-complete chunk files automatically)"
echo "  - Then resubmit phase 3 manually:"
echo "      sbatch 03_merge_and_analyze.slurm"
