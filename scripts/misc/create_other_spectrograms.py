#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os
# import autorootcwd
USER = os.getenv("USER")
EDDIE = os.getenv("EDDIE")

REPO_HOME = f"/home/{USER}/LeadSheetTranscription"
DATA_HOME = f"{EDDIE}/data/processed"

input_dir = f"{DATA_HOME}/audio"
output_dir = f"{DATA_HOME}/cache/4096"

chroma = f"python {REPO_HOME}/src/data/create_chroma_cqts.py --input_dir={input_dir} --output_dir={output_dir}/chroma_cqts"

output_file = open("./scripts/experiments.txt", "w")
print(chroma, file=output_file)

linear = f"python {REPO_HOME}/src/data/create_linear_spectrograms.py --input_dir={input_dir} --output_dir={output_dir}/linear"
print(linear, file=output_file)

mel = f"python {REPO_HOME}/src/data/create_melspectrograms.py --input_dir={input_dir} --output_dir={output_dir}/mels"
print(mel, file=output_file)

output_file.close()
