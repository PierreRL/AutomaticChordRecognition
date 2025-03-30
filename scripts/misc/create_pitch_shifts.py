#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

USER = os.getenv("USER")

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"

input_dir = f"{REPO_HOME}/data/processed/audio"

base_call = f"python {REPO_HOME}/src/data/create_pitch_shifted_audio.py --output_dir={REPO_HOME}/data/processed/audio/augs"

output_file = open("./scripts/experiments.txt", "w")

for f in os.listdir(input_dir):
    if not f.endswith(".mp3"):
        continue
    
    expt_call = (
        f"{base_call} --file={f}"
    )

    print(expt_call, file=output_file)

output_file.close()
