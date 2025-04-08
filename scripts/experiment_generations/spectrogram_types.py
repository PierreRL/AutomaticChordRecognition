#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv("USER")
EDDIE = os.getenv("EDDIE")

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"
DATA_HOME = f"{EDDIE}/data/processed"

base_call = f"python {REPO_HOME}/src/run.py --input_dir={DATA_HOME} --output_dir={DATA_HOME}/experiments/spectrogram_types"

spectrogram_types = ["cqt", "mel", "linear", "chroma"]

output_file = open("./scripts/experiments.txt", "w")

for spectrogram in spectrogram_types:
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} --exp_name={spectrogram} --spectrogram_type={spectrogram}"
    )
    print(expt_call, file=output_file)

output_file.close()
