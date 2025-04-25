#!/bin/bash
#SBATCH --job-name=pinn         # Job name
#SBATCH --output=logs/output_%j.log    # Output file (%j = job ID)
#SBATCH --error=logs/error_%j.log      # Error file
    #SBATCH --ntasks=1                # Number of tasks (processes)
#SBATCH --cpus-per-task=1         # Number of CPUs per task
#SBATCH --time=12:00:00           # Time limit hrs:min:sec

# Load necessary modules (if needed)
# module load python/3.9

# Print job info
echo "Running job on $(hostname)"
echo "Starting at $(date)"

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate fair-chem 
# Run your program
# cd ~/protein_protein_docking
# boltz predict /home/dyang20/protein_protein_docking/eval/fastas/holo_decoys/7tvh__B1_Q9I2Q1--7tvh__A1_Q9I2Q1.fasta  --use_msa_server --out_dir /home/dyang20/protein_protein_docking/eval/boltz1/raw/holo_decoys --output_format pdb
python /home/dyang20/EScAIP/main.py \
  --mode train \
  --config-yml /home/dyang20/EScAIP/configs/s2ef/MD22/EScAIP/DHA.yml \
  --run-dir /home/dyang20/EScAIP

echo "Finished at $(date)"

