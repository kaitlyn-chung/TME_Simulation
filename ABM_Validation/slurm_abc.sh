#!/bin/bash
# sbatch slurm_abc.sh   -> one patient per array task (edit --array if the patient count changes)
#SBATCH --job-name=abc_tme
#SBATCH --array=0
#SBATCH --cpus-per-task=100
#SBATCH --mem=32G
#SBATCH --partition=scu-cpu
#SBATCH --output=/athena/cayuga_0014/scratch/kch4018/abm-files/ABM_Validation/logs/abc_%A_%a.out

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/athena/cayuga_0014/scratch/kch4018/abm-files/TME_Simulation    # folder that contains the ABM_Jul25 package
mkdir -p logs

smc_abc="/athena/cayuga_0014/scratch/kch4018/abm-files/ABM_Validation/abc_smc_tme.py"

# If the job hits the time limit, just resubmit: it continues from abc_results/<patient>/abc.db
python -u $smc_abc $SLURM_ARRAY_TASK_ID