#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv("USER")

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"
DATA_HOME = f"{REPO_HOME}/data/processed"

base_call = f"python {REPO_HOME}/src/run.py --input_dir={DATA_HOME}"

output_file = open("./scripts/experiments.txt", "w")

args_set = [
    f"--weight_loss --output_dir={REPO_HOME}/experiments/weight_loss_on ",
    f" --output_dir={REPO_HOME}/experiments/weight_loss_off ",
]
n = 10

for args in args_set:
    for rep in range(n):
        # Note that we don't set a seed for rep - a seed is selected at random
        # and recorded in the output data by the python script
        expt_call = f"{base_call} " + args
        print(expt_call, file=output_file)

output_file.close()
