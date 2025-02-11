#!/bin/bash
#SBATCH --job-name=run_experiment_%j
#SBATCH --output=logs/run_experiment_%j.log
#SBATCH --error=logs/run_experiment_%j.err
#
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

# Set up the virtual environment (only if not already created)
VENV_DIR=~/ug4_venv
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip install --upgrade pip
    pip install -r ~/LeadSheetTranscription/requirements.txt
else
    source $VENV_DIR/bin/activate
fi

cd ~/LeadSheetTranscription

# Run your Python script
python src/run.py --exp_name='testing_slurm'

# Deactivate the virtual environment (optional but good practice)
deactivate