#!/usr/bin/env python
"""
Script that shifts chord IDs by -5 to +6 semitones for a given tensor files.
"""
import autorootcwd
import os
import argparse
from tqdm import tqdm

import torch
import numpy as np

from src.utils import transpose_chord_id_vector

def shift_chords(input_dir: str, file_name: str, output_dir: str):
    """
    Shift chord IDs by -5 to +6 semitones for a single tensor file. Saves each transposition.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.basename(file_name)
    file_path = os.path.join(input_dir, file_name)

    # Create pitch shifts from -5 to +6 semitones
    for semitone in range(-5, 7):
        if semitone == 0:
            continue

        chord_ids = torch.load(file_path)
        # If file already exists, skip
        output_file_name = f"{os.path.splitext(file_name)[0]}_shifted_{semitone}.pt"
        output_file_path = os.path.join(output_dir, output_file_name)
        if os.path.isfile(output_file_path):
            print(f"File {output_file_path} already exists. Skipping.")
            continue
        
        # Shift using transposition function
        shifted_chord_ids = transpose_chord_id_vector(chord_ids, semitone)

        # Convert to tensor
        shifted_chord_ids = torch.tensor(shifted_chord_ids, dtype=torch.long)

        # Save the shifted chord IDs
        torch.save(shifted_chord_ids, output_file_path)

def pitch_shift_chords(
        input_dir: str, 
        output_dir: str, 
        start_idx: int = None, 
        end_idx: int = None
    ):
    """
    Process either a single file or all MP3 files in a directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Process all MP3 files in the input directory
    files = [f for f in os.listdir(input_dir) if f.endswith(".pt")]
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(files)
    
    files = files[start_idx:end_idx]
    for file_name in tqdm(files, desc="Processing files"):
        shift_chords(input_dir, file_name, output_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Create pitch-shifted audios for MP3 files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input audio files (MP3)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the pitch-shifted audio files."
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="Start index for processing files."
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index for processing files."
    )
    args = parser.parse_args()

    # Process all MP3 files in the input directory
    pitch_shift_chords(
        args.input_dir,
        args.output_dir,
        args.start_idx,
        args.end_idx
    )

if __name__ == "__main__":
    main()