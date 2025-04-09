#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

USER = os.getenv("USER")
EDDIE = os.getenv("EDDIE")

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"
DATA_HOME = f"{EDDIE}/data/processed"

input_dir = f"{DATA_HOME}/audio"
files = os.listdir(input_dir)
files = [f for f in files if f.endswith(".mp3")]
max_group_size = 25
if len(files) % max_group_size != 0:
    print(
        f"Warning: {len(files)} files in {input_dir} not divisible by {max_group_size}. "
        "This may lead to uneven distribution of files across groups."
    )

base_call = f"python {REPO_HOME}/src/data/cache_beats.py --input_dir={input_dir} --output_dir={DATA_HOME}/cache/beats"

output_file = open("./scripts/experiments.txt", "w")


for i in range(0, len(files), max_group_size):
    start_idx = i
    end_idx = min(i + max_group_size, len(files))

    expt_call = (
        f"{base_call} --start_idx={start_idx} --end_idx={end_idx}"
    )

    print(expt_call, file=output_file)


output_file.close()
