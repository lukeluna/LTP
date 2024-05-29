#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --gpus-per-node=1
#SBATCH --job-name=LTP-Mistral
#SBATCH --mem=4GB
#SBATCH --partition=gpu

# Clear the module environment
module purge
# Load the Python version that has been used to construct the virtual environment
# we are using below
module load Python/3.11.3-GCCcore-12.3.0

# Activate the virtual environment
source ~/virtual_env/jupyter_tunnel/bin/activate

# Start the jupyter server, using the hostname of the node as the way to connect to it
jupyter notebook --no-browser --ip=$( hostname )