#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv("USER")
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = "/disk/scratch"
SCRATCH_HOME = f"{SCRATCH_DISK}/{USER}"

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"
DATA_HOME = f"{REPO_HOME}/data/processed"

base_call = f"python {REPO_HOME}/src/run.py --input_dir={DATA_HOME} --output_dir={REPO_HOME}/experiments/large_vocab"
output_file = open("./scripts/experiments.txt", "w")

expt_call = (
    f"{base_call} "
    f"--hidden_size=256 --segment_length=10 --enable_early_stopping --epochs=100 --lr_scheduler=plateau --lr=0.001 --exp_name=large-vocab-defaults"
)
print(expt_call, file=output_file)

output_file.close()
