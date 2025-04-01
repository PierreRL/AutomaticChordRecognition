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
#$ -l h_vmem=24G

# Request the number of CPUs per task (equivalent to `--cpus-per-task` in SLURM)
#$ -pe smp 1


# Set the maximum runtime for the job (similar to SLURM's `--time`)
#$ -l h_rt=02:00:00

# Specify which queue to use, for example:
#$ -q all.q

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
experiment_text_file=$1
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