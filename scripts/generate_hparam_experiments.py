#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv("USER")
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = "/disk/scratch"
SCRATCH_HOME = f"{SCRATCH_DISK}/{USER}"

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"
DATA_HOME = f"{REPO_HOME}/data/processed"

base_call = f"python {REPO_HOME}/src/run.py --input_dir={DATA_HOME} --output_dir={SCRATCH_HOME}/experiments/hparams --no_early_stopping"

segment_lengths = [10, 20, 30, 60]
layers = [1, 2, 3]
hidden_sizes = [128, 256, 512]
params = [
    (segment_length, layer, hidden_size)
    for segment_length in segment_lengths
    for layer in layers
    for hidden_size in hidden_sizes
]
nr_expts = len(params)

nr_servers = 10
avg_expt_time = 20  # mins
print(f"Total experiments = {nr_expts}")
print(f"Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs")

output_file = open("./scripts/experiments.txt", "w")

for segment_length, layer, hidden_size in params:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} "
        f"--segment_length={segment_length} "
        f"--num_layers={layer} "
        f"--hidden_size={hidden_size} "
        f"--exp_name=segment_length_{segment_length}_layers_{layer}_hidden_size_{hidden_size}"
    )
    print(expt_call, file=output_file)

output_file.close()
