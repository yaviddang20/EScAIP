#!/bin/bash
#SBATCH --job-name=pinn         # Job name
#SBATCH --output=logs/output_%j.log    # Output file (%j = job ID)
#SBATCH --error=logs/error_%j.log      # Error file
#SBATCH --ntasks=1                # Number of tasks (processes)
#SBATCH --cpus-per-task=1         # Number of CPUs per task
#SBATCH --time=12:00:00           # Time limit hrs:min:sec
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpu
#SBATCH --nodelist=germain

echo "Running job on $(hostname)"
echo "Starting at $(date)"

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate fair-chem

export WANDB_API_KEY=1b6fa9bc95f59bc410efdfc6778345c1367ff626

python /home/dyang20/EScAIP/main.py \
  --mode train \
  --config-yml /home/dyang20/EScAIP/configs/s2ef/MD22/EScAIP/DHA.yml \
  --run-dir /home/dyang20/EScAIP

echo "Finished at $(date)"

