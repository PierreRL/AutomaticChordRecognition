#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

USER = os.getenv("USER")
EDDIE = os.getenv("EDDIE")
REPO_HOME = f"/home/{USER}/LeadSheetTranscription"

if EDDIE is None:
    EDDIE = REPO_HOME  # Compatibility with eddie and mlt clusters

DATA_HOME = f"{EDDIE}/data/processed"

input_dir = f"{DATA_HOME}/audio"
files = os.listdir(input_dir)
files = [f for f in files if f.endswith(".mp3")]
max_group_size = 1300
base_call = f"python {REPO_HOME}/src/data/generative_features/create_generative_features_jukebox.py --dir={DATA_HOME}"

output_file = open("./scripts/experiments_jukebox_gens.txt", "w")


for i in range(0, len(files), max_group_size):
    start_idx = i
    end_idx = min(i + max_group_size, len(files))

    expt_call = (
        f"{base_call} --start_idx={start_idx} --end_idx={end_idx}"
    )

    print(expt_call, file=output_file)


output_file.close()
