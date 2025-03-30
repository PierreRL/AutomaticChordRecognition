#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv("USER")

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"
DATA_HOME = f"{REPO_HOME}/data/processed"

base_call = f"python {REPO_HOME}/src/run.py --input_dir={DATA_HOME} --output_dir={REPO_HOME}/experiments/pitch_shifts --segment_length=10"

output_file = open("./scripts/experiments.txt", "w")

expt_call = (
    f"{base_call} --exp_name=baseline"
)
print(expt_call, file=output_file)

expt_call = (
    f"{base_call} --exp_name=cqt_shift --cqt_pitch_shift"
)
print(expt_call, file=output_file)

expt_call = (
    f"{base_call} --exp_name=audio_shift --audio_pitch_shift"
)
print(expt_call, file=output_file)

output_file.close()
