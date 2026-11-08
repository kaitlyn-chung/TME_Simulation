#!/bin/bash

#SBATCH --job-name=clear
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=scu-cpu

rm -rf /athena/cayuga_0014/scratch/kch4018/abm-files/TME_Simulation/ABM_Jul25/logs
rm -rf /athena/cayuga_0014/scratch/kch4018/abm-files/TME_Simulation/ABM_Jul25/sensitivity_results

mkdir -p /athena/cayuga_0014/scratch/kch4018/abm-files/TME_Simulation/ABM_Jul25/sensitivity_results