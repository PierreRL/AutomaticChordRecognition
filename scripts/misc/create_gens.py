#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

USER = os.getenv("USER")

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"
DATA_HOME = f"{REPO_HOME}/data/processed"

base_call = f"python {REPO_HOME}/src/data/create_generative_features.py --dir={DATA_HOME}"

output_file = open("./scripts/experiments.txt", "w")

print(base_call, file=output_file)

output_file.close()
