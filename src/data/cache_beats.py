import autorootcwd
import os
import argparse
from tqdm import tqdm
import numpy as np
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

def main(
    input_dir="data/processed/audio",
    output_dir="data/processed/cache/beats",
    start_idx=None,
    end_idx=None,
):
    os.makedirs(output_dir, exist_ok=True)

    filenames = os.listdir(input_dir)
    filenames = [f for f in filenames if f.endswith(".mp3")]
    filenames = [os.path.splitext(f)[0] for f in filenames]
    filenames.sort()

    if start_idx is not None and end_idx is not None:
        filenames = filenames[start_idx:end_idx]
    elif start_idx is not None:
        filenames = filenames[start_idx:]
    elif end_idx is not None:
        filenames = filenames[:end_idx]

    act_proc = RNNBeatProcessor()
    beat_proc = DBNBeatTrackingProcessor(fps=100)

    for filename in tqdm(filenames):
        input_path = os.path.join(input_dir, f"{filename}.mp3")
        output_path = os.path.join(output_dir, f"{filename}.npy")

        if os.path.exists(output_path):
            continue

        try:
            activations = act_proc(input_path)
            beats = beat_proc(activations)
            np.save(output_path, beats)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache beat timings using madmom.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/processed/audio",
        help="Directory containing input audio files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/cache/beats",
        help="Directory to save cached beat times.",
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
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
