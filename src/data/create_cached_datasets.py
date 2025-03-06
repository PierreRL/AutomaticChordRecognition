import autorootcwd
import os
import torch
import argparse
from tqdm import tqdm
from src.utils import get_cqt, get_chord_annotation, get_filenames, SR
from src.data.dataset import FullChordDataset


def main(
    hop_length, create_cqts, input_dir="data/processed", output_dir="data/processed"
):
    dataset = FullChordDataset(hop_length=hop_length, input_dir=output_dir)
    os.makedirs(dataset.cqt_cache_dir, exist_ok=True)
    os.makedirs(dataset.chord_cache_dir, exist_ok=True)

    filenames = get_filenames()
    for filename in tqdm(filenames):
        if create_cqts:
            cqt = get_cqt(
                filename, hop_length=dataset.hop_length, override_dir=input_dir
            )
            torch.save(cqt, f"{dataset.cqt_cache_dir}/{filename}.pt")

        chord_ids = get_chord_annotation(
            filename,
            frame_length=hop_length / SR,
            override_dir=input_dir,
        )
        torch.save(chord_ids, f"{dataset.chord_cache_dir}/{filename}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache dataset for faster loading.")
    parser.add_argument(
        "--hop_length", type=int, required=True, help="Hop length to use in processing."
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
    args = parser.parse_args()

    main(args.hop_length, args.create_cqts)
