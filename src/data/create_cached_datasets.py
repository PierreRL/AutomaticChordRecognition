"""
Script to create cached datasets for faster loading.

Creates cqt and chord id tensors. The script should be run twice with the --create_cqts and --create_chords flags respectively. These should be stored in the folder structure expected by the FullChordDataset class.
"""

import autorootcwd
import os
import torch
import argparse
from tqdm import tqdm
from src.utils import get_cqt, get_chord_annotation, get_synthetic_annotation, SR


def main(
    create_cqts=True,
    hop_length=4096,
    input_dir="data/processed/audio",
    output_dir="data/processed/cache",
    create_chords=True,
    start_idx=None,
    end_idx=None,
    synthetic=False,
):
    os.makedirs(output_dir, exist_ok=True)

    filenames = os.listdir(input_dir)


    # Filter filenames based on the type of data being created
    if create_cqts:
        if synthetic:
            filenames = [f for f in filenames if f.endswith(".wav")]
        else:
            filenames = [f for f in filenames if f.endswith(".mp3")]
    elif create_chords:
        if synthetic:
            filenames = [f for f in filenames if f.endswith(".json")]
        else:
            filenames = [f for f in filenames if f.endswith(".jams")]

    filenames = [os.path.splitext(f)[0] for f in filenames]
    
    if start_idx is not None and end_idx is not None:
        filenames = filenames[start_idx:end_idx]
    elif start_idx is not None:
        filenames = filenames[start_idx:]
    elif end_idx is not None:
        filenames = filenames[:end_idx]

    for filename in tqdm(filenames):
        if create_cqts:
            if os.path.exists(f"{output_dir}/{filename}.pt"):
                continue
            if synthetic:
                file_extension = "wav"
            else:
                file_extension = "mp3"
            cqt = get_cqt(filename, hop_length=hop_length, override_dir=input_dir, file_extension=file_extension)
            torch.save(cqt, f"{output_dir}/{filename}.pt")

        if create_chords:
            if synthetic:
                chord_ids = get_synthetic_annotation(
                    filename,
                    frame_length=hop_length / SR,
                    override_dir=input_dir,
                )
            else:
                chord_ids = get_chord_annotation(
                    filename,
                    frame_length=hop_length / SR,
                    override_dir=input_dir,
                )
            torch.save(chord_ids, f"{output_dir}/{filename}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache dataset for faster loading.")
    parser.add_argument(
        "--hop_length",
        type=int,
        required=False,
        help="Hop length to use in processing.",
        default=4096,
    )
    parser.add_argument(
        "--create_cqts", action="store_true", help="Flag to create CQTs (takes ~1hr)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/processed",
        help="Directory containing the processed data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save the cached data.",
    )
    parser.add_argument(
        "--create_chords",
        action="store_true",
        help="Flag to create the cached chord annotations.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="Start index for processing files.",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index for processing files.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Flag to create synthetic data.",
    )
    args = parser.parse_args()

    main(
        create_cqts=args.create_cqts,
        hop_length=args.hop_length,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        create_chords=args.create_chords,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        synthetic=args.synthetic,
    )
