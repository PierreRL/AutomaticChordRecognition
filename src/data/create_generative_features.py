"""
Script to create generative features for the dataset using MusicGen.
"""

import autorootcwd

import torch
import os
import argparse
from tqdm import tqdm

from src.data.dataset import FullChordDataset
from src.utils import get_filenames, get_torch_device, SR
from src.data.musicgen import get_musicgen_model, extract_song_hidden_representation


def main(
    hop_length=4096,
    dir = "data/processed",
    model_size="large",
    max_chunk_length=5
):
    device = get_torch_device(allow_mps=False)
    print(f"Using device: {device}")
    dataset = FullChordDataset(hop_length=hop_length, input_dir=dir, generative_features=True)
    os.makedirs(dataset.gen_cache_dir, exist_ok=True)

    filenames = get_filenames(directory=f"{dir}/audio")
    print('Loading model...')
    model = get_musicgen_model(model_size=model_size, device=device)
    frame_length = hop_length / SR

    print('Extracting features...')
    for filename in tqdm(filenames):
        song_repr = extract_song_hidden_representation(
            dir=dir,
            filename=filename,
            max_chunk_length=max_chunk_length,
            model=model,
            frame_length=frame_length
        )
        torch.save(song_repr, f"{dataset.gen_cache_dir}/{filename}.pt")


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
        "--dir",
        type=str,
        default="data/processed",
        help="Directory for input and output. Expects audio files in dir/audio and will output to dir/gen.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="large",
        help="Size of the model to use. Can be 'small' or 'large'.",
    )
    parser.add_argument(
        "--max_chunk_length",
        type=float,
        default=10,
        help="The length of context in seconds to pass through the model at once. Absolute maximum 30s."
    )
    args = parser.parse_args()

    main(
        hop_length=args.hop_length,
        dir=args.dir,
        model_size=args.model_size,
        max_chunk_length=args.max_chunk_length
    )
