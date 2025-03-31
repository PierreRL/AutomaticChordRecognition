#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv("USER")

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"
DATA_HOME = f"{REPO_HOME}/data/processed"

base_call = f"python {REPO_HOME}/src/run.py --input_dir={DATA_HOME} --output_dir={REPO_HOME}/experiments/gen_layers "

gen_layers = list(range(49,1,-1))

output_file = open("./scripts/experiments.txt", "w")

for layer in gen_layers:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} --exp_name=layer_{layer} --gen_layer={layer} --no_cqt --use_generative_features --segment_length=10"
    )
    print(expt_call, file=output_file)

output_file.close()
