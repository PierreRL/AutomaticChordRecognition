#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv("USER")
EDDIE = os.getenv("EDDIE")

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"
DATA_HOME = f"{EDDIE}/data/processed"

base_call = f"python {REPO_HOME}/src/run.py --input_dir={DATA_HOME} --output_dir={EDDIE}/experiments/smoothers"
output_file = open("./scripts/experiments.txt", "w")

expt_call = (
    f"{base_call} --exp_name=no_smoothing --no_hmm_smoothing"
)
print(expt_call, file=output_file)

expt_call = (
    f"{base_call} --exp_name=hmm_smoothing"
)
print(expt_call, file=output_file)

expt_call = (
    f"{base_call} --exp_name=crf_smoothing --no_hmm_smoothing --crf"
)
print(expt_call, file=output_file)

output_file.close()
