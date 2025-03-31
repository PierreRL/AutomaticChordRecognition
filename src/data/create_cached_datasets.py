import autorootcwd
import os
import torch
import argparse
from tqdm import tqdm
from src.utils import get_cqt, get_chord_annotation, get_filenames, SR
from src.data.dataset import FullChordDataset


def main(
    create_cqts=True,
    hop_length=4096,
    input_dir="data/processed",
    output_dir="data/processed",
    create_chords=True,
    start_idx=None,
    end_idx=None,
):
    os.makedirs(output_dir, exist_ok=True)

    filenames = get_filenames(dir=input_dir)
    if start_idx is not None and end_idx is not None:
        filenames = filenames[start_idx:end_idx]
    elif start_idx is not None:
        filenames = filenames[start_idx:]
    elif end_idx is not None:
        filenames = filenames[:end_idx]

    for filename in tqdm(filenames):
        if create_cqts:
            cqt = get_cqt(
                filename, hop_length=hop_length, override_dir=input_dir
            )
            torch.save(cqt, f"{output_dir}/{filename}.pt")

        if create_chords:
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
    args = parser.parse_args()

    main(
        create_cqts=args.create_cqts,
        hop_length=args.hop_length,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        create_chords=args.create_chords,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
