#!/bin/bash
# ====================
# Options for sbatch
# ====================
#SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH --error=/home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

echo "Starting job $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NODELIST"
echo "Job submitted at `date`"
echo "-----------------------------------"

echo "Setting up bash enviroment"
source ~/.bashrc
set -e

# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}


# Set up the virtual environment
echo "Loading modules"
source ~/ug4_venv/bin/activate


# Move data
echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

repo_home=/home/${USER}/LeadSheetTranscription
src_path=${repo_home}/data/processed

# input data directory path on the scratch disk of the node
dest_path=${SCRATCH_HOME}/data/processed
mkdir -p ${dest_path}  # make it if required

rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

echo "Running script"
cd ${repo_home}

experiment_text_file=$1
COMMAND="`sed '1q;d' ${experiment_text_file}`"
echo "Running command from file: ${COMMAND}"
eval "${COMMAND}"
python /home/s2147950/LeadSheetTranscription/src/data/create_generative_features.py --dir=/home/s2147950/LeadSheetTranscription/data/processed --start_idx=519 --end_idx=520 --output_dir=${SCRATCH_HOME}/gen-large

# ======================================
# Move output data from scratch to DFS
# ======================================

echo "Moving output data back to DFS"

src_path=${SCRATCH_HOME}/gen-large
dest_path=${repo_home}/gen_from_node
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

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