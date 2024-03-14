#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --account=protein-lm-scaling
#SBATCH --partition=a40x
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --output=./slurm/id_analysis_plm-%j.out
#SBATCH --error=./slurm/id_analysis_plm-%j.err

source /weka/home-talkhanz/plm_scaling_venv/bin/activate
srun python /admin/home-talkhanz/repos/forks/talkhanz/latest/protein-lm-scaling/scripts/analysis.py id_analysis