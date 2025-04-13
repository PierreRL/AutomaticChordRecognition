import autorootcwd
import argparse
import os
from tqdm import tqdm
import numpy as np
import torch

from src.utils import audio_write, write_json
from src.data.generative_features.musicgen import get_musicgen_model
from src.data.synthetic_data.descriptions import sample_description
from src.data.synthetic_data.chord_sequence import sample_chord_sequence

# Import the MusiConGen model and utility for writing audio.
from MusiConGen.audiocraft.audiocraft.models import MusicGen as MusiConGen


def generate_song(
    model: MusiConGen,
    song_length: int = 220,
    bpm_mean: float = 117,
    bpm_std: float = 28,
):

    # Sample a BPM value from a normal distribution, ensuring it's within a realistic range.
    bpm = int(np.random.normal(bpm_mean, bpm_std))
    while bpm < 60 or bpm > 200:
        bpm = int(np.random.normal(bpm_mean, bpm_std))

    # Define a default time signature (using 4 for 4/4 time)
    meter = 4

    # Sample a text description and chord sequence from external functions.
    description = sample_description()
    num_beats = int(bpm * song_length / 60)  # Total number of beats in the song
    chord_seq = sample_chord_sequence(seq_length=num_beats)

    # Set the generation parameters.
    # If the song length is greater than the model's base duration (e.g., 30 sec),
    # use an extend_stride to generate in overlapping segments.
    extend_stride = 15 if song_length > 30 else 0
    model.set_generation_params(
        duration=song_length, extend_stride=extend_stride, top_k=250
    )

    metadata = {
        "description": description,
        "bpm": bpm,
        "meter": meter,
        "chord_sequence": chord_seq,
    }

    # Generate the audio conditioned on description, chord sequence, BPM, and meter.
    audio_out = model.generate_with_chords_and_beats(
        [description],  # list of descriptions (one per sample)
        [chord_seq],  # list of chord progressions
        [bpm],  # list of BPM values
        [meter],  # list of meter values (e.g., 4 for 4/4)
    )
    return audio_out.cpu(), model.sample_rate, metadata


def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'metadata'), exist_ok=True)

    print(f"Generating {args.num_songs} songs...")
    # Ensure the output directory exists.
    os.makedirs(args.output_dir, exist_ok=True)
    # Set the random seed for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading the MusiConGen model...")

    # Load the pretrained MusiConGen model.
    model = get_musicgen_model(model_name="chord", device="cuda")

    for song_idx in tqdm(range(args.num_songs), desc="Generating songs"):
        # Generate the song audio and sample rate.
        audio_tensor, sample_rate, metadata = generate_song(
            model, args.song_length, args.bpm_mean, args.bpm_std
        )

        # Define the output file name.
        output_file = os.path.join(args.output_dir, 'audio', f"synthetic_{song_idx+1}.wav")

        # Write the audio to a WAV file with loudness normalization.
        audio_write(output_file, audio_tensor, sample_rate)
        print(f"Saved generated song to: {output_file}")

        # Write metdata
        write_json(metadata, os.path.join(args.output_dir, 'metadata', f"metadata_{song_idx+1}.json"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate songs with MusiConGen.")
    parser.add_argument(
        "--num_songs", type=int, default=1, help="Number of songs to generate."
    )
    parser.add_argument(
        "--song_length", type=int, default=220, help="Length for each song in seconds."
    )
    parser.add_argument(
        "--bpm_mean", type=float, default=117.0, help="Mean BPM for sampling."
    )
    parser.add_argument(
        "--bpm_std",
        type=float,
        default=28.0,
        help="Standard deviation for BPM sampling.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/generated_songs",
        help="Directory to save generated WAV files.",
    )

    args = parser.parse_args()
    main(args)
