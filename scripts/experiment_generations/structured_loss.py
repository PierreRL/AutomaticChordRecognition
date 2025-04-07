#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv("USER")
EDDIE = os.getenv("EDDIE")

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"
DATA_HOME = f"{EDDIE}/data/processed"

base_call = f"python {REPO_HOME}/src/run.py --input_dir={DATA_HOME} --output_dir={EDDIE}/experiments/pitch_shifts --structured_loss"

output_file = open("./scripts/experiments.txt", "w")

alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for alpha in alphas:
    expt_call = (
        f"{base_call} --exp_name=alpha_{alpha} --structured_loss_alpha={alpha}"
    )
    print(expt_call, file=output_file)

output_file.close()