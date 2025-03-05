#!/bin/bash

# Define hop lengths to iterate ovter
hop_lengths=(512 1024 2048 8192 16384)

# Path to the script
script_path="./src/data/create_cached_datasets.py"  # Update this if your script has a different name

for hop_length in "${hop_lengths[@]}"; do
    echo "Running script with hop_length=$hop_length..."
    python "$script_path" --hop_length "$hop_length" --create_cqts
done