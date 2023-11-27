#!/bin/bash 
#SBATCH --job-name=Importance
#SBATCH -o ./logs/Importance-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=27

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

echo the job id is $SLURM_JOB_ID

python3 -u perm_importance_grads.py