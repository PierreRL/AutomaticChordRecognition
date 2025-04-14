#!/bin/bash
# ====================
# Options for sbatch
# ====================

#SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH --error=/home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00

echo "Starting job $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NODELIST"
echo "Job submitted at `date`"
echo "-----------------------------------"

# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}

# Setup environment
echo "Loading modules"

source ~/jukebox_venv/bin/activate

# Load any modules you need
echo "Loading bash source"
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc


# Move data
# echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

repo_home=/home/${USER}/LeadSheetTranscription
# src_path=${repo_home}/data/processed

# Set up any environment variables
export HF_HUB_OFFLINE=1

# # input data directory path on the scratch disk of the node
# dest_path=${SCRATCH_HOME}/data/processed
# mkdir -p ${dest_path}  # make it if required

# rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

# # Empty the output directory
# # rm -rf ${SCRATCH_HOME}/experiments
# mkdir -p ${SCRATCH_HOME}/experiments

echo "Running script"
cd ${repo_home}

# Run script
experiment_text_file=$1
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d \" ${experiment_text_file}`"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"

# ======================================
# Move output data from scratch to DFS
# ======================================
# This presumes your command wrote data to some known directory. In this
# example, send it back to the DFS with rsync

echo "Moving output data back to DFS"

# src_path=${SCRATCH_HOME}/experiments
# dest_path=${repo_home}/experiments/
# rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

# Clean up the node's scratch disk
# rm -r ${SCRATCH_HOME}


# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"

# Deactivate the virtual environment (optional but good practice)
deactivate