#!/usr/bin/env python
"""
Script that creates pitch-shifted audios by -5 to +6 semitones for a given audio file.
If no file is specified, it processes all MP3 files in the input directory.
Optionally keeps stereo or converts to mono.
"""

import os
import argparse
from tqdm import tqdm

import pyrubberband
import torchaudio
import torch
import numpy as np

def pitch_shift_file(input_dir: str, file_name: str, output_dir: str, keep_stereo: bool):
    """
    Create pitch-shifted audios by -5 to +6 semitones for a single audio file.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.basename(file_name)
    file_path = os.path.join(input_dir, file_name)
    waveform, sample_rate = torchaudio.load(file_path)

    # If not keeping stereo, convert to mono (averaging channels)
    if not keep_stereo:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Convert to numpy array with shape (samples, channels)
    waveform_np = waveform.numpy().T

    # Create pitch shifts from -5 to +6 semitones
    for semitone in range(-5, 7):
        if semitone == 0:
            continue
        shifted = pyrubberband.pitch_shift(waveform_np, sample_rate, semitone)
        # Convert back to (channels, samples) for torchaudio.save
        shifted_tensor = torch.tensor(shifted.T)
        
        output_file_name = f"{os.path.splitext(file_name)[0]}_shifted_{semitone}.mp3"
        output_file_path = os.path.join(output_dir, output_file_name)
        torchaudio.save(output_file_path, shifted_tensor, sample_rate)

# def create_pitch_shifted_audios(input_dir: str, output_dir: str, keep_stereo: bool, single_file: str = None):
#     """
#     Process either a single file or all MP3 files in a directory.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     if single_file:
#         file_path = os.path.join(input_dir, single_file)
#         if not os.path.isfile(file_path):
#             print(f"Error: {file_path} does not exist.")
#             return
#         print(f"Processing single file: {file_path}")
#         pitch_shift_file(file_path, output_dir, keep_stereo)
#     else:
#         # Process all MP3 files in the input directory
#         files = [f for f in os.listdir(input_dir) if f.endswith(".mp3")]
#         for file_name in tqdm(files, desc="Processing files"):
#             file_path = os.path.join(input_dir, file_name)
#             pitch_shift_file(file_path, output_dir, keep_stereo)

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
        "--mono",
        action="store_true",
        help="Convert audio to mono. By default, the script keeps the original stereo channels."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Process only this file (filename relative to input_dir) instead of all files."
    )
    args = parser.parse_args()

    # create_pitch_shifted_audios(args.input_dir, args.output_dir, keep_stereo=not args.mono, single_file=args.file)
    pitch_shift_file(args.input_dir, args.file, args.output_dir, keep_stereo=not args.mono)

if __name__ == "__main__":
    main()