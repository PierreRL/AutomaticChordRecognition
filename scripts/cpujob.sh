#!/bin/bash
# ====================
# Options for Grid Engine
# ====================
#$ -cwd
#$ -o /home/$USER/logs/job-$JOB_ID-$TASK_ID.out
#$ -e /home/$USER/logs/job-$JOB_ID-$TASK_ID.out

#$ -l h_vmem=16G
#$ -l h_rt=03:00:00

# ===================
# Environment Setup
# ===================

echo "Starting job $JOB_ID"
echo "Running on $HOSTNAME"
echo "Job submitted at $(date)"
echo "-----------------------------------"

echo "Setting up bash environment"
source ~/.bashrc
set -e

REPO_HOME="/home/${USER}/LeadSheetTranscription"
DATA_HOME="/exports/eddie/scratch/s2147950/"

# Activate virtual environment
echo "Loading virtual environment"
source "${DATA_HOME}/diss_venv/bin/activate"

# ===================
# Job Execution
# ===================

echo "Running experiment script"
cd ${REPO_HOME}

# Read the experiment command from the experiments file
python src/run.py --input_dir=${DATA_HOME}/data/processed --output_dir=${DATA_HOME}/experiments

# Print the command to be run
echo "Running provided command: ${COMMAND}"

# Execute the command
eval "${COMMAND}"

echo "Command ran successfully!"
echo ""
echo "============"
echo "Job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"

deactivate