#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv("USER")

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"
DATA_HOME = f"{REPO_HOME}/data/processed"

base_call = f"python {REPO_HOME}/src/run.py --input_dir={DATA_HOME} --output_dir={REPO_HOME}/experiments/gen_layers "

gen_reductions = ['avg', 'concat', 'codebook_0' , 'codebook_1', 'codebook_2', 'codebook_3']

output_file = open("./scripts/experiments.txt", "w")

for reduction in gen_reductions:
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} --exp_name=reduction_{reduction} --gen_reduction={reduction} "
    )
    print(expt_call, file=output_file)

output_file.close()
