"""
This script is used to generate the train, val test, splits for the dataset.

It takes in all the filenames, splits at random at given ratios and saves the splits in a json file.
"""

import autorootcwd
import os
import json
import random

from src.utils import get_filenames

SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}
OUT_DIR = "data/processed"


def generate_splits():
    random.seed(42)
    # Get all the filenames
    filenames = get_filenames()

    # Shuffle the filenames
    random.shuffle(filenames)

    # Get the number of files
    num_files = len(filenames)

    # Get the number of files in each split
    num_train_files = int(SPLIT_RATIOS["train"] * num_files)
    num_val_files = int(SPLIT_RATIOS["val"] * num_files)

    # Split the filenames
    train_files = filenames[:num_train_files]
    val_files = filenames[num_train_files : num_train_files + num_val_files]
    test_files = filenames[num_train_files + num_val_files :]

    # Save the splits
    splits = {"train": train_files, "val": val_files, "test": test_files}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/splits.json", "w") as f:
        json.dump(splits, f, indent=4)


if __name__ == "__main__":
    generate_splits()
