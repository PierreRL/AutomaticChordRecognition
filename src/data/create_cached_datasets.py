"""
Creates a cached version of dataset.

This script is run once to prepare the data to be faster to load.

The cached version of the dataset is stored in the following directories:
- data/processed/cqt
- data/processed/chord_ids
"""

import autorootcwd
import os
import torch
from tqdm import tqdm
from src.utils import get_cqt, get_chord_annotation, get_filenames
from src.data.dataset import FullChordDataset

# Set to True to create the cached CQTs (takes ~1hr)
create_cqts = False


def main():
    dataset = FullChordDataset()
    os.makedirs(dataset.cqt_cache_dir, exist_ok=True)
    os.makedirs(dataset.chord_cache_dir, exist_ok=True)

    filenames = get_filenames()
    for filename in tqdm(filenames):
        if create_cqts:
            cqt = get_cqt(
                filename,
                sr=dataset.sr,
                hop_length=dataset.hop_length,
                n_bins=dataset.n_bins,
                bins_per_octave=dataset.bins_per_octave,
            )
            torch.save(cqt, f"{dataset.cqt_cache_dir}/{filename}.pt")

        chord_one_hot = get_chord_annotation(
            filename,
            frame_length=dataset.hop_length / dataset.sr,
        )
        torch.save(chord_one_hot, f"{dataset.chord_cache_dir}/{filename}.pt")


if __name__ == "__main__":
    main()
