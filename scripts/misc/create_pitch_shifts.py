#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

USER = os.getenv("USER")

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"

base_call = f"python {REPO_HOME}/src/data/create_pitch_shifted_audios.py --input_dir={REPO_HOME}/data/processed/audio --output_dir={REPO_HOME}/data/processed/audio"

output_file = open("./scripts/experiments.txt", "w")

print(base_call, file=output_file)

output_file.close()
