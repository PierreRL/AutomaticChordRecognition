#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv("USER")

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"
DATA_HOME = f"{REPO_HOME}/data/processed"

base_call = f"python {REPO_HOME}/src/run.py --input_dir={DATA_HOME} --output_dir={REPO_HOME}/experiments/pitch_shifts --segment_length=10"

output_file = open("./scripts/experiments.txt", "w")

# expt_call = (
#     f"{base_call} --exp_name=baseline"
# )
# print(expt_call, file=output_file)

# expt_call = (
#     f"{base_call} --exp_name=cqt_shift --cqt_pitch_shift"
# )
# print(expt_call, file=output_file)

aug_probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for prob in aug_probs:
    expt_call = (
        f"{base_call} --exp_name=audio_shift_{prob} --audio_pitch_shift --aug_shift_prob={prob}"
    )
    print(expt_call, file=output_file)

output_file.close()
