#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --account=protein-lm-scaling
#SBATCH --partition=a40x
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --output=./slurm/plm-%j.out
#SBATCH --error=./slurm/plm-%j.err

source /weka/home-talkhanz/plm_scaling_venv/bin/activate
srun python /admin/home-talkhanz/repos/forks/talkhanz/protein-lm-scaling/scripts/analysis.py target_seq