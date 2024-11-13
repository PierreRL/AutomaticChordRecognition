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
from src.utils import get_cqt, chord_ann_to_tensor, get_filenames
from src.data.dataset import ChordDataset


def main():

    dataset = ChordDataset()
    os.makedirs(dataset.cqt_cache_dir, exist_ok=True)
    os.makedirs(dataset.chord_cache_dir, exist_ok=True)

    sr = 22050
    hop_length = 2048
    n_bins = 24 * 6
    bins_per_octave = 24

    filenames = get_filenames()
    for filename in tqdm(filenames):
        cqt = get_cqt(
            filename,
            sr=sr,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
        chord_one_hot = chord_ann_to_tensor(filename, frame_length=hop_length / sr)

        torch.save(cqt, f"{dataset.cqt_cache_dir}/{filename}.pt")
        torch.save(chord_one_hot, f"{dataset.chord_cache_dir}/{filename}.pt")


if __name__ == "__main__":
    main()
