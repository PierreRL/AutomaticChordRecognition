import autorootcwd
import os
import torch
import argparse
from tqdm import tqdm
from src.utils import get_chroma_cqt

def main(
    hop_length=4096,
    input_dir="data/processed/audio",
    output_dir="data/processed/cache/4096/chroma_cqt",
    start_idx=None,
    end_idx=None,
):
    os.makedirs(output_dir, exist_ok=True)

    filenames = os.listdir(input_dir)
    filenames = [f for f in filenames if f.endswith(".mp3")]
    filenames = [os.path.splitext(f)[0] for f in filenames]
    if start_idx is not None and end_idx is not None:
        filenames = filenames[start_idx:end_idx]
    elif start_idx is not None:
        filenames = filenames[start_idx:]
    elif end_idx is not None:
        filenames = filenames[:end_idx]

    for filename in tqdm(filenames):
        if os.path.exists(f"{output_dir}/{filename}.pt"):
            continue
        cqt = get_chroma_cqt(
            filename, hop_length=hop_length, override_dir=input_dir
        )
        torch.save(cqt, f"{output_dir}/{filename}.pt")


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
        "--input_dir",
        type=str,
        default="data/processed/audio",
        help="Directory containing the processed data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/cache/4096/chroma_cqt",
        help="Directory to save the cached data.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="Start index for processing files.",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index for processing files.",
    )
    args = parser.parse_args()

    main(
        hop_length=args.hop_length,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
