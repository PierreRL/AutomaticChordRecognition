#!/usr/bin/env python
import os
import argparse
import torch
from tqdm import tqdm


def reconstruct_from_concat(
    data_dir: str,
    hop_length: int,
    model_name: str,
    K: int,
):
    """
    Reconstruct 'avg' and 'codebook_<k>' representations from the 'concat' PT files.

    For each '.pt' file in {data_dir}/cache/{hop_length}/gen-{model_name}/concat/:
      - Load the [T, K*card] concat tensor
      - Reshape into [T, K, card]
      - Compute:
        * avg: mean over the K dimension -> [T, card]
        * codebook_k: each [T, card]
      - Save each representation under:
          {data_dir}/cache/{hop_length}/gen-{model_name}/avg/
          {data_dir}/cache/{hop_length}/gen-{model_name}/codebook_k/
    """
    concat_dir = os.path.join(
        data_dir, "cache", str(hop_length), f"gen-{model_name}", "concat"
    )
    avg_dir = os.path.join(
        data_dir, "cache", str(hop_length), f"gen-{model_name}", "avg"
    )
    codebook_dirs = [
        os.path.join(
            data_dir, "cache", str(hop_length), f"gen-{model_name}", f"codebook_{k}"
        )
        for k in range(K)
    ]

    # Create output folders
    os.makedirs(avg_dir, exist_ok=True)
    for cdir in codebook_dirs:
        os.makedirs(cdir, exist_ok=True)

    # List all .pt files in the "concat" folder
    if not os.path.isdir(concat_dir):
        raise ValueError(f"Concat directory not found: {concat_dir}")
    all_concat_files = [fn for fn in os.listdir(concat_dir) if fn.endswith(".pt")]

    print(f"Found {len(all_concat_files)} 'concat' files in {concat_dir}")

    for concat_file in tqdm(all_concat_files, desc="Reconstructing"):
        concat_path = os.path.join(concat_dir, concat_file)
        avg_path = os.path.join(avg_dir, concat_file)
        codebook_paths = [os.path.join(codebook_dirs[k], concat_file) for k in range(K)]

        # Load the concat representation
        concat_tensor = torch.load(concat_path, map_location="cpu")
        T, concat_dim = concat_tensor.shape
        if concat_dim % K != 0:
            raise ValueError(
                f"Concat dimension ({concat_dim}) must be divisible by K ({K})!"
            )

        card = concat_dim // K
        stacked = concat_tensor.reshape(T, K, card)  # shape: [T, K, card]

        # Compute avg over codebooks -> shape: [T, card]
        avg_rep = stacked.mean(dim=1)

        # Extract each codebook -> shape: [T, card]
        codebook_reps = [stacked[:, k, :].contiguous() for k in range(K)]

        # Save average representation
        torch.save(avg_rep, avg_path)

        # Save codebook representations
        for k in range(K):
            torch.save(codebook_reps[k], codebook_paths[k])


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct avg/codebook representations from 'concat' PT files."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to the main directory containing 'cache' subfolders.",
    )
    parser.add_argument(
        "--hop_length", type=int, default=4096, help="Hop length in subfolder naming."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="large",
        help="Model size in subfolder naming, e.g. 'small'.",
    )
    parser.add_argument("--K", type=int, default=4, help="Number of codebooks.")
    args = parser.parse_args()

    reconstruct_from_concat(
        data_dir=args.dir,
        hop_length=args.hop_length,
        model_name=args.model_name,
        K=args.K,
    )


if __name__ == "__main__":
    main()
