"""
Script to create generative features for the dataset using MusicGen.
"""

import autorootcwd

import os
import argparse
from tqdm import tqdm

import torch
from data.generative_features.jukebox_features import get_jukebox_model, extract_song_hidden_representation_jukebox
SR = 44100

def main(
    hop_length=4096,
    dir="./data/processed",
    output_dir=None,
    start_idx=0,
    end_idx=None,
    max_chunk_length=5,
    batch_size=16,
):
    if output_dir is None:
        output_dir = f"{dir}/cache/{hop_length}/gen-jukebox"
    device = get_torch_device(allow_mps=False)
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    filenames = get_filenames(dir=f"{dir}/audio")
    print("Loading model...")
    bundle = get_jukebox_model()

    frame_length = hop_length / SR

    if end_idx is None:
        end_idx = len(filenames)
    if start_idx > end_idx:
        raise ValueError("start_idx must be less than or equal to end_idx")
    if start_idx is None:
        start_idx = 0

    filenames = filenames[start_idx:end_idx]

    print("Extracting features...")
    for filename in tqdm(filenames):
        # Skip if the file already exists
        if os.path.exists(f"{output_dir}/codebook_0/{filename}.pt"):
            print(f"Skipping {filename} as it already exists.")
            continue
        song_repr_dict = extract_song_hidden_representation_jukebox(
            dir=dir,
            filename=filename,
            max_chunk_length=max_chunk_length,
            jukebox_bundle=bundle,
            frame_length=frame_length,
            max_batch_size=batch_size,
        )
        for reduction, song_repr in song_repr_dict.items():
            os.makedirs(f"{output_dir}/{reduction}", exist_ok=True)
            torch.save(song_repr, f"{output_dir}/{reduction}/{filename}.pt")
            # Free up memory
            del song_repr
        del song_repr_dict
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


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
        "--output_dir",
        type=str,
        default=None,
        help="Optional different directory for output. If not provided, will use dir/gen.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="large",
        help="Size of the model to use. Can be 'small' or 'large'.",
    )
    parser.add_argument(
        "--max_chunk_length",
        type=float,
        default=10,
        help="The length of context in seconds to pass through the model at once. Absolute maximum 30s.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for processing. Default is 16.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        required=False,
        default=0,
        help="Start index for the filenames. Useful for parallel processing.",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        required=False,
        default=None,
        help="End index for the filenames. Useful for parallel processing.",
    )
    args = parser.parse_args()

    main(
        hop_length=args.hop_length,
        dir=args.dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_chunk_length=args.max_chunk_length,
        batch_size=args.batch_size,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )

def get_torch_device(allow_mps=False):
    """
    Get the torch device to use for training.

    Returns:
        torch.device: The torch device to use.
    """

    # CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")

    # MPS
    if allow_mps and torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS is available but not built.")
        return torch.device("mps")

    # CPU
    return torch.device("cpu")

def get_filenames(dir: str = "data/processed/audio") -> list:
    """
    Get a list of filenames in a directory.

    Args:
        directory (str): The directory to get the filenames from.

    Returns:
        filenames (list): A list of filenames in the directory.
    """
    filenames = os.listdir(dir)
    filenames = [
        filename.split(".")[0] for filename in filenames if filename.endswith(".mp3")
    ]
    return filenames