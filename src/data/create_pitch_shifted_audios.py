#!/usr/bin/env python
"""
Script that creates pitch-shifted audios by -5 to +6 semitones for all audio files in a directory.
Optionally keeps stereo or converts to mono.
"""

import os
import argparse
from tqdm import tqdm

import pyrubberband
import torchaudio
import torch
import numpy as np

def create_pitch_shifted_audios(input_dir: str, output_dir: str, keep_stereo: bool):
    """
    Create pitch-shifted audios by -5 to +6 semitones for all audio files in a directory.

    Parameters:
      input_dir (str): Directory containing input audio files.
      output_dir (str): Directory where pitch-shifted files will be saved.
      keep_stereo (bool): If True, keep the original channel layout; if False, convert to mono.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Only process .mp3 files
    files = [f for f in os.listdir(input_dir) if f.endswith(".mp3")]
    for file_name in tqdm(files, desc="Processing files"):
        file_path = os.path.join(input_dir, file_name)
        waveform, sample_rate = torchaudio.load(file_path)  # shape: (channels, samples)

        # If not keeping stereo, convert to mono (averaging channels)
        if not keep_stereo:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Convert to numpy array with shape (samples, channels)
        waveform_np = waveform.numpy().T

        # Create pitch shifts from -5 to +6 semitones
        for semitone in range(-5, 7):
            shifted = pyrubberband.pitch_shift(waveform_np, sample_rate, semitone)
            # Convert back to (channels, samples) for torchaudio.save
            shifted_tensor = torch.tensor(shifted.T)
            
            output_file_name = f"{os.path.splitext(file_name)[0]}_shifted_{semitone}.mp3"
            output_file_path = os.path.join(output_dir, output_file_name)
            torchaudio.save(output_file_path, shifted_tensor, sample_rate)

def main():
    parser = argparse.ArgumentParser(
        description="Create pitch-shifted audios for all MP3 files in a directory."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        default="./data/processed/audio",
        help="Directory containing input audio files (MP3)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        default="./data/processed/audio",
        help="Directory to save the pitch-shifted audio files."
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Convert audio to mono. By default, the script keeps the original stereo channels."
    )
    args = parser.parse_args()

    create_pitch_shifted_audios(args.input_dir, args.output_dir, keep_stereo=not args.mono)

if __name__ == "__main__":
    main()