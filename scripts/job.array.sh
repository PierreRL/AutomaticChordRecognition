#!/bin/bash
# ====================
# Options for Grid Engine
# ====================

# Specify the working directory for the job to run from
#$ -cwd

# Log location for standard output and error (similar to SLURM output and error)
#$ -o /home/$USER/gridengine_logs/job-$JOB_ID.out
#$ -e /home/$USER/gridengine_logs/job-$JOB_ID.err

# Request memory (in MB, equivalent to `--mem` in SLURM)
#$ -l h_vmem=16G

# Request GPU 
#$ -l gpu=1  # Request one GPU

# Set the maximum runtime for the job (similar to SLURM's `--time`)
#$ -l h_rt=02:00:00

# ===================
# Environment Setup
# ===================

echo "Starting job $JOB_ID"
echo "Running on $HOSTNAME"
echo "Job submitted at $(date)"
echo "-----------------------------------"

# Load your environment (if needed, adjust based on your setup)
echo "Setting up bash environment"
source ~/.bashrc

# Make the script exit after the first error (equivalent to `set -e` in bash)
set -e

# Setup scratch space (if needed for large data)
$REPO_HOME="/home/${USER}/LeadSheetTranscription"
$DATA_HOME="/exports/eddie/scratch/s2147950/"

# Activate virtual environment
echo "Loading virtual environment"
source "${DATA_HOME}/diss_venv/bin/activate"

# ===================
# Job Execution
# ===================

echo "Running experiment script"
cd ${REPO_HOME}

# Read the experiment command from the experiments file
experiment_text_file="${REPO_HOME}/scripts/experiments.txt"
if [ ! -f "${experiment_text_file}" ]; then
    echo "Error: Experiment text file not found at ${experiment_text_file}"
    exit 1
fi
if [ -z "${SGE_TASK_ID}" ]; then
    echo "Error: SGE_TASK_ID is not set. Please submit the job as an array job."
    exit 1
fi
COMMAND=$(sed -n "${SGE_TASK_ID}p" ${experiment_text_file})

# Print the command to be run
echo "Running provided command: ${COMMAND}"

# Execute the command
eval "${COMMAND}"

echo "Command ran successfully!"

# ======================
# Post-Experiment Logging
# ======================
echo ""
echo "============"
echo "Job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"

# Deactivate virtual environment (optional)
deactivate