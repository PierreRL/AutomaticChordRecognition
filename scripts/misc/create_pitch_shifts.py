#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os
import autorootcwd
USER = os.getenv("USER")

# REPO_HOME = f"/home/{USER}/LeadSheetTranscription"

input_dir = f"./data/processed/audio"
output_dir = f"./data/processed/audio/augs"
os.makedirs(output_dir, exist_ok=True)
files = os.listdir(input_dir)
files = [f for f in files if f.endswith(".mp3")]
max_group_size = 10
if len(files) % max_group_size != 0:
    print(
        f"Warning: {len(files)} files in {input_dir} not divisible by {max_group_size}. "
        "This may lead to uneven distribution of files across groups."
    )


base_call = f"python src/data/create_pitch_shifted_audio.py --input_dir={input_dir} --output_dir={output_dir}"

output_file = open("./scripts/experiments.txt", "w")

for i in range(0, len(files), max_group_size):
    start_idx = i
    end_idx = min(i + max_group_size, len(files))

    expt_call = (
        f"{base_call} "
        f"--start_idx={start_idx} --end_idx={end_idx}"
    )

    print(expt_call, file=output_file)

output_file.close()
