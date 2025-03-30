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
    dir = "./data/processed",
    model_size="large",
    layer_indices=None,
    max_chunk_length=5
):
    device = get_torch_device(allow_mps=False)
    print(f"Using device: {device}")
    dataset = FullChordDataset(hop_length=hop_length, input_dir=dir)
    os.makedirs(dataset.gen_cache_dir, exist_ok=True)

    filenames = get_filenames(dir=f"{dir}/audio")
    print('Loading model...')
    model = get_musicgen_model(model_size=model_size, device=device)
    frame_length = hop_length / SR

    for layer_idx in layer_indices:
        os.makedirs(f"{dir}/cache/gen/{layer_idx}", exist_ok=True)

    print('Extracting features...')
    for filename in tqdm(filenames):
        song_repr_dict = extract_song_hidden_representation(
            dir=dir,
            filename=filename,
            max_chunk_length=max_chunk_length,
            model=model,
            frame_length=frame_length,
            layer_indices=layer_indices
        )
        for layer_idx, song_repr in song_repr_dict.items():
            # Save each layer's representation separately
            torch.save(song_repr, f"{dir}/cache/{hop_length}/gen/{layer_idx}/{filename}.pt")


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
        default="./data/processed",
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
    parser.add_argument(
        "--layer_indices",
        type=str,
        required=False,
        default=None,
        help="Comma-separated list of layer indices to extract. If None, uses the final layer only."
    )
    args = parser.parse_args()

    # Parse layer indices, e.g. "10,18" => [10, 18]
    if args.layer_indices is not None:
        layer_indices = [int(x.strip()) for x in args.layer_indices.split(',')]
    else:
        layer_indices = list(range(1,49))  # Default to all layers


    main(
        hop_length=args.hop_length,
        dir=args.dir,
        model_size=args.model_size,
        max_chunk_length=args.max_chunk_length,
        layer_indices=layer_indices,
    )
