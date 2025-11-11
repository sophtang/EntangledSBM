#!/bin/bash

#SBATCH --job-name=trpcage-entangled
#SBATCH --output=/path/to/your/home/EntangledSBM/entangled-tps/logs/trpcage_%j.out
#SBATCH --error=/path/to/your/home/EntangledSBM/entangled-tps/logs/trpcage_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1

# Load any required modules (adjust based on your cluster setup)
# module load cuda/11.8
# module load anaconda3

# Set working directory
cd /path/to/your/home/EntangledSBM/entangled-tps

# Create logs directory if it doesn't exist
mkdir -p logs

# Make the script executable
chmod +x scripts/trpcage.bash

# Run the training script
./scripts/trpcage.bash

echo "Job completed at $(date)"