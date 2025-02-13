import autorootcwd
import os
import torch
import argparse
from tqdm import tqdm
from src.utils import get_cqt, get_chord_annotation, get_filenames, SR
from src.data.dataset import FullChordDataset


def main(hop_length, create_cqts):
    dataset = FullChordDataset(hop_length=hop_length)
    os.makedirs(dataset.cqt_cache_dir, exist_ok=True)
    os.makedirs(dataset.chord_cache_dir, exist_ok=True)

    filenames = get_filenames()
    for filename in tqdm(filenames):
        if create_cqts:
            cqt = get_cqt(filename, dataset.hop_length)
            torch.save(cqt, f"{dataset.cqt_cache_dir}/{filename}.pt")

        chord_one_hot = get_chord_annotation(
            filename,
            frame_length=hop_length / SR,
        )
        torch.save(chord_one_hot, f"{dataset.chord_cache_dir}/{filename}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache dataset for faster loading.")
    parser.add_argument(
        "--hop_length", type=int, required=True, help="Hop length to use in processing."
    )
    parser.add_argument(
        "--create_cqts", action="store_true", help="Flag to create CQTs (takes ~1hr)"
    )
    args = parser.parse_args()

    main(args.hop_length, args.create_cqts)
