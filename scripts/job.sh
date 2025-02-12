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

echo "Loading modules"
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


echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

# input data directory path on the DFS - change line below if loc different
repo_home=/home/${USER}/LeadSheetTranscription
src_path=${repo_home}/data/processed

# input data directory path on the scratch disk of the node
dest_path=${SCRATCH_HOME}/data/processed
mkdir -p ${dest_path}  # make it if required

rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

echo "Running script"

# Run your Python script
python ${repo_home}/src/run.py --exp_name='testing_slurm' --input_dir=${dest_path} --output_dir=${SCRATCH_HOME}/experiments --epochs=10

# ======================================
# Move output data from scratch to DFS
# ======================================
# This presumes your command wrote data to some known directory. In this
# example, send it back to the DFS with rsync

echo "Moving output data back to DFS"

src_path=${SCRATCH_HOME}/experiments
dest_path=${repo_home}/experiments/

# Deactivate the virtual environment (optional but good practice)
deactivate

# Clean up the node's scratch disk
rm -r ${SCRATCH_HOME}

# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"