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
    ignore_chords=False,
):
    dataset = FullChordDataset(hop_length=hop_length, input_dir=output_dir)
    os.makedirs(dataset.cqt_cache_dir, exist_ok=True)
    os.makedirs(dataset.chord_cache_dir, exist_ok=True)

    filenames = get_filenames(directory=f"{input_dir}/audio")
    for filename in tqdm(filenames):
        if create_cqts:
            cqt = get_cqt(
                filename, hop_length=dataset.hop_length, override_dir=input_dir
            )
            torch.save(cqt, f"{dataset.cqt_cache_dir}/{filename}.pt")

        if ignore_chords:
            continue
        chord_ids = get_chord_annotation(
            filename,
            frame_length=hop_length / SR,
            override_dir=input_dir,
        )
        torch.save(chord_ids, f"{dataset.chord_cache_dir}/{filename}.pt")


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
        "--ignore_chords",
        action="store_true",
        help="Flag to ignore the chord annotations.",
    )
    args = parser.parse_args()

    main(
        create_cqts=args.create_cqts,
        hop_length=args.hop_length,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ignore_chords=args.ignore_chords,
    )
