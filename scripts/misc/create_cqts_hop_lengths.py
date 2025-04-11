#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv("USER")

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"
DATA_HOME = f"{REPO_HOME}/data/processed"

base_call = f"python {REPO_HOME}/src/data/create_cached_datasets.py --input_dir={DATA_HOME}/audio"

hop_lengths = [512, 1024, 2048, 4096, 8192, 16384]
nr_expts = len(hop_lengths)

nr_servers = 10
avg_expt_time = 60  # mins
print(f"Total experiments = {nr_expts}")
print(f"Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs")

output_file = open("./scripts/experiments.txt", "w")

for hop_length in hop_lengths:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} "
        f"--hop_length={hop_length} --create_cqts --output_dir={DATA_HOME}/cache/{hop_length}"
    )
    print(expt_call, file=output_file)

output_file.close()
