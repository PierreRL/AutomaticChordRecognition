#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv("USER")
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = "/disk/scratch"
SCRATCH_HOME = f"{SCRATCH_DISK}/{USER}"

REPO_HOME = f"/home/${USER}/LeadSheetTranscription"
DATA_HOME = f"{REPO_HOME}/data/processed"

base_call = f"python {REPO_HOME}/src/run.py --input_dir={DATA_HOME}/input --output_dir={SCRATCH_HOME}/experiments "

learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
schedulers = ["cosine", "plateau", "none"]
params = [(lr, sched) for lr in learning_rates for sched in schedulers]
nr_expts = len(learning_rates) * len(schedulers)

nr_servers = 10
avg_expt_time = 20  # mins
print(f"Total experiments = {nr_expts}")
print(f"Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs")

output_file = open("./scripts/experiments.txt", "w")

for lr, sched in params:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} "
        f"--lr={lr} --lr_scheduler={sched} --exp_name=lr-search_{sched}_{lr}"
    )
    print(expt_call, file=output_file)

output_file.close()
