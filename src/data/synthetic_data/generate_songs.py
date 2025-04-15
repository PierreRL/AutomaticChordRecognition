import autorootcwd
import argparse
import os
from tqdm import tqdm
import numpy as np
import torch

from src.utils import audio_write, write_json
from src.data.generative_features.musicgen import get_musicgen_model
from src.data.synthetic_data.descriptions import generate_description
from src.data.synthetic_data.chord_sequence import generate_jazz_progression, reformat_chord_sequence

from MusiConGen.audiocraft.audiocraft.models import MusicGen as MusiConGen

def generate_batch(model: MusiConGen, batch_size: int, song_length: int, bpm_mean: float, bpm_std: float):
    bpm_list = []
    description_list = []
    chord_seq_list = []
    meter_list = []
    metadata_list = []

    for _ in range(batch_size):
        bpm = int(np.random.normal(bpm_mean, bpm_std))
        while bpm < 60 or bpm > 200:
            bpm = int(np.random.normal(bpm_mean, bpm_std))

        description = generate_description()
        meter = 4
        chord_seq = generate_jazz_progression()

        bpm_list.append(bpm)
        description_list.append(description)
        chord_seq_list.append(chord_seq)
        meter_list.append(meter)

        metadata = {
            "description": description,
            "bpm": bpm,
            "meter": meter,
            "chord_sequence": chord_seq,
        }
        metadata_list.append(metadata)
    model.set_generation_params(
        duration=song_length, extend_stride=15, top_k=250
    )

    audio_out_batch = model.generate_with_chords_and_beats(
        description_list,
        chord_seq_list,
        bpm_list,
        meter_list
    )
    return audio_out_batch.cpu(), model.sample_rate, metadata_list



def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'metadata'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'chords'), exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading the MusiConGen model...")
    model = get_musicgen_model(model_name="chord", device="cuda")

    total_batches = (args.num_songs + args.batch_size - 1) // args.batch_size
    song_idx = args.start_idx
    song_length = 30  # seconds

    for _ in tqdm(range(total_batches), desc="Generating batches"):
        batch_size = min(args.batch_size, args.num_songs - song_idx)
        audio_batch, sample_rate, metadata_batch = generate_batch(
            model=model, 
            batch_size=batch_size,
            song_length=song_length,
            bpm_mean=args.bpm_mean,
            bpm_std=args.bpm_std, 
        )

        for i in range(batch_size):
            output_file = os.path.join(args.output_dir, 'audio', f"synthetic_{song_idx}.wav")
            audio_write(output_file, audio_batch[i], sample_rate)
            write_json(metadata_batch[i], os.path.join(args.output_dir, 'metadata', f"metadata_{song_idx}.json"))
            chord_seq = reformat_chord_sequence(metadata_batch[i], song_length=song_length)
            write_json(chord_seq, os.path.join(args.output_dir, 'chords', f"chords_{song_idx}.json"))
            print(f"Saved song {song_idx} to {output_file}")
            song_idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate songs with MusiConGen in batches.")
    parser.add_argument("--num_songs", type=int, default=1, help="Number of songs to generate.")
    # parser.add_argument("--song_length", type=int, default=30, help="Length for each song in seconds.")
    parser.add_argument("--bpm_mean", type=float, default=117.0, help="Mean BPM for sampling.")
    parser.add_argument("--bpm_std", type=float, default=28.0, help="Standard deviation for BPM sampling.")
    parser.add_argument("--output_dir", type=str, default="./data/synthetic_songs", help="Directory to save generated WAV files.")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for song naming.")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of songs to generate per batch.")
    args = parser.parse_args()
    main(args)
