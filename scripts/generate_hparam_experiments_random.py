#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os
import numpy as np

# The home dir on the node's scratch disk
USER = os.getenv("USER")
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = "/disk/scratch"
SCRATCH_HOME = f"{SCRATCH_DISK}/{USER}"

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"
DATA_HOME = f"{REPO_HOME}/data/processed"

base_call = f"python {REPO_HOME}/src/run.py --input_dir={DATA_HOME} --output_dir={SCRATCH_HOME}/experiments/hparams_random --no_early_stopping"

segment_range = [10, 60]
layers = [1, 2, 3]
hidden_size_range = [64, 512]
nr_expts = 32
params = []
for i in range(nr_expts):
    segment_length = np.random.randint(segment_range[0], segment_range[1])
    layer = np.random.choice(layers)
    hidden_size = np.random.randint(hidden_size_range[0], hidden_size_range[1])
    params.append((segment_length, layer, hidden_size))

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
