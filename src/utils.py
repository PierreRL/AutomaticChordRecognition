"""
Various utility functions for processing music data.
"""

import autorootcwd
import os
import math
import random
import json
import numpy as np
import torch
from functools import lru_cache
from typing import List, Tuple

# Music processing libraries
import librosa
import jams
from harte.harte import Harte

# Constants
SMALL_VOCABULARY = True
SR = 44100
HOP_LENGTH = 4096
BINS_PER_OCTAVE = 36
N_BINS = BINS_PER_OCTAVE * 6

if SMALL_VOCABULARY:
    NUM_CHORDS = 26
else:
    NUM_CHORDS = 170


# Functions
def get_filenames(directory: str = "data/processed/audio") -> list:
    """
    Get a list of filenames in a directory.

    Args:
        directory (str): The directory to get the filenames from.

    Returns:
        filenames (list): A list of filenames in the directory.
    """
    filenames = os.listdir(directory)
    filenames = [
        filename.split(".")[0] for filename in filenames if filename.endswith(".mp3")
    ]
    return filenames


def get_split_filenames() -> Tuple[List, List, List]:
    """Get the filenames for the train, validation, and test sets.

    Returns:

        train_filenames (list): The filenames for the training set.
        val_filenames (list): The filenames for the validation set.
        test_filenames (list): The filenames for the test set."""
    with open("data/processed/splits.json", "r") as f:
        splits = json.load(f)

    return splits["train"], splits["val"], splits["test"]


@lru_cache(maxsize=None)
def get_raw_chord_annotation(filename, override_dir=None):
    """
    Retrieves the raw chord annotation data from a JAMS file.

    Args:
        filename (str): The filename of the JAMS file to load.

    Returns:
        chord_annotation (SortedKeyDict): An ordered dict of chord annotations.
    """
    if override_dir is not None:
        jam = jams.load(os.path.join(f"{override_dir}/chords/", f"{filename}.jams"))
    else:
        jam = jams.load(os.path.join("./data/processed/chords/", f"{filename}.jams"))
    chord_ann = jam.annotations.search(namespace="chord")[0]
    return chord_ann.data


@lru_cache(maxsize=None)
def get_annotation_metadata(filename):
    """
    Retrieves the metadata from a JAMS file.

    Args:
        filename (str): The filename of the JAMS file to load.

    Returns:
        metadata (dict): The metadata of the JAMS file.
    """
    jam = jams.load(os.path.join("./data/processed/chords/", f"{filename}.jams"))
    metadata = jam.file_metadata
    return metadata


def get_cqt(
    filename: str,
    override_dir: str = None,
    sr: int = SR,
    hop_length: int = HOP_LENGTH,
    n_bins: int = N_BINS,
    bins_per_octave: int = BINS_PER_OCTAVE,
    fmin: float = librosa.note_to_hz("C1"),
    absolute: bool = True,
    return_as_tensor: bool = True,
) -> torch.Tensor:
    """
    Compute the log CQT of an audio file.

    Args:
        filename (str): The filename of the audio file to load.
        sr (int): The sample rate of the audio file.
        hop_length (int): The hop length of the CQT.
        n_bins (int): The number of bins in the CQT.
        bins_per_octave (int): The number of bins per octave in the CQT.
        fmin (float): The minimum frequency of the CQT.

    Returns:
        cqt (torch.Tensor): The log CQT of the audio file. Has shape (num_frames, n_bins).
    """
    # Default hyperparameters from https://brianmcfee.net/papers/ismir2017_chord.pdf
    if override_dir is not None:
        src = librosa.load(
            os.path.join(f"{override_dir}/audio", f"{filename}.mp3"), sr=sr
        )[0]
    else:
        src = librosa.load(
            os.path.join("./data/processed/audio/", f"{filename}.mp3"), sr=sr
        )[0]

    cqt = librosa.cqt(
        src,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
    )
    if absolute:
        cqt = np.abs(cqt)

    cqt = librosa.amplitude_to_db(cqt)

    if return_as_tensor:
        return torch.tensor(cqt, dtype=torch.float32).T
    return cqt.T


def pitch_shift_cqt(
    cqt: torch.Tensor, semitones: int, bins_per_octave=BINS_PER_OCTAVE
) -> torch.Tensor:
    """
    Apply a pitch shift to a log CQT.

    Args:
        cqt (torch.Tensor): The log CQT to pitch shift. Has shape (num_frames, n_bins).
        semitones (int): The number of semitones to shift the audio.
        sr (int): The sample rate of the audio file.

    Returns:
        cqt_shifted (torch.Tensor): The pitch shifted log CQT. Has shape (num_frames, n_bins).
    """

    # Compute the number of bins to shift
    bin_shift = int(semitones * bins_per_octave / 12)

    # Create an empty tensor to hold the shifted CQT with -80 dB fill value
    shifted_cqt = torch.full_like(cqt, fill_value=-80)

    # Shift the bins
    if bin_shift > 0:
        shifted_cqt[:, bin_shift:] = cqt[:, :-bin_shift]
    elif bin_shift < 0:
        shifted_cqt[:, :bin_shift] = cqt[:, -bin_shift:]
    else:
        shifted_cqt = cqt.clone()

    return shifted_cqt


def cqt_to_audio(
    cqt: torch.Tensor,
    sr: int = SR,
    hop_length: int = HOP_LENGTH,
    bins_per_octave: int = 36,
):
    """
    Convert a CQT to audio using the inverse CQT.

    Args:
        cqt (torch.Tensor): The CQT (in Amplitude) to convert to audio. If given in dB, convert to amplitude first.
        sr (int): The sample rate of the audio file.
        hop_length (int): The hop length of the CQT.
        bins_per_octave (int): The number of bins per octave in the CQT.

    Returns:
        audio (np.ndarray): The audio signal reconstructed from the CQT.
    """

    # Convert the CQT to a numpy array and reshape to (n_bins, num_frames)
    cqt = cqt.T.numpy()

    # Invert the CQT to a waveform
    audio = librosa.icqt(
        cqt, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave
    )

    return audio


@lru_cache(maxsize=None)
def get_pitch_classes(chord_str: str):
    """
    Extracts pitch classes from the chord string.
    Converts the string to a Harte object internally and computes pitch classes.

    Args:
        chord_str (str): The chord in Harte string format.

    Returns:
        tuple: A sorted tuple of pitch classes (integers 0-11).
    """
    # Parse the chord using the Harte library
    chord = Harte(chord_str)

    # Return the pitch classes as a sorted tuple
    prime_form_chord = chord.normalOrder

    # Transpose the chord to C
    root = chord.root().pitchClass
    prime_form_chord = [(note - root) % 12 for note in prime_form_chord]

    # Remove duplicates
    prime_form_chord = list(set(prime_form_chord))

    return tuple(sorted(prime_form_chord))


@lru_cache(maxsize=None)
def chord_to_id(chord: str, use_small_vocab: bool = SMALL_VOCABULARY) -> int:
    """
    Convert a chord to an index corresponding to a chord id.

    Args:
        chord (str): Harte string representation of a chord.
        use_small_vocab (bool): If True, use the small vocabulary; otherwise, use the large vocabulary.

    Returns:
        torch.Tensor: The chord id.
            - For small vocab: id in the range 0-24 (0 is reserved for N, X).
            - For large vocab: id in the range 0-170 (0 for N, 1 for X, 2-170 for root and quality combinations).
    """
    # If the chord is N or X, return the corresponding id (0 for N, 1 for X in large vocab)
    if chord == "N":
        return 0
    if chord == "X":
        return 1

    # Parse the chord using the Harte library
    try:
        parsed_chord = Harte(chord)
    except Exception:
        # Raise an error if the chord is invalid
        raise ValueError(f"Invalid chord format: {chord}")

    if use_small_vocab:
        # Small vocabulary (0-24)
        root = parsed_chord.root().pitchClass
        quality = parsed_chord.quality
        if quality == "major":
            quality = 0
        elif quality == "minor":
            quality = 1
        else:
            return 1  # Map unknown chords to X

        return root + 12 * quality + 2

    else:
        # Large vocabulary (0-170)
        root = parsed_chord.root().pitchClass
        pitch_classes = get_pitch_classes(chord)

        # Define the templates
        templates = {
            (0, 4, 7): "maj",
            (0, 3, 7): "min",
            (0, 3, 6): "dim",
            (0, 4, 8): "aug",
            (0, 3, 7, 9): "min6",
            (0, 4, 7, 9): "maj6",
            (0, 3, 7, 10): "min7",
            (0, 3, 7, 11): "minmaj7",
            (0, 4, 7, 11): "maj7",
            (0, 4, 7, 10): "7",
            (0, 3, 6, 9): "dim7",
            (0, 3, 6, 10): "hdim7",
            (0, 2, 7): "sus2",
            (0, 5, 7): "sus4",
        }

        quality = templates.get(pitch_classes, None)
        if quality is None:
            quality = parsed_chord.quality
            if quality == "other":
                return 1  # Map unknown chords to X
            else:
                quality = quality[
                    :3
                ]  # Get the first 3 characters of the quality 'min', 'maj', 'dim' or 'aug'

        # Compute id based on root and quality
        quality_index = list(templates.values()).index(quality)
        return 2 + root * 14 + quality_index  # Offset by 2 for N and X


@lru_cache(maxsize=None)
def get_chord_root(id: int, use_small_vocab: bool = SMALL_VOCABULARY) -> str:
    """
    Get the root of a chord id.

    Args:
        id (int): The chord id.
        use_small_vocab (bool): If True, use the small vocabulary; otherwise, use the large vocabulary.

    Returns:
        str: The root of the chord.
    """
    if use_small_vocab:
        # Small vocabulary (0-24)
        if id < 0 or id > 25:
            raise ValueError("Chord id must be in the range 0-24.")

        if id == 0:
            return "N"

        if id == 1:
            return "X"

        root_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        root = root_name[(id - 2) % 12]

        return root

    else:
        # Large vocabulary (0-170)
        if id < 0 or id > 170:
            raise ValueError("Chord id must be in the range 0-170.")

        if id == 0:
            return "N"
        elif id == 1:
            return "X"

        # Subtract 2 to account for N and X
        id -= 2

        root = id // 14

        root_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        root = root_name[root]

        return root


@lru_cache(maxsize=None)
def get_chord_quality(id: int, use_small_vocab: bool = SMALL_VOCABULARY) -> str:
    """
    Get the quality of a chord id.

    Args:
        id (int): The chord id.
        use_small_vocab (bool): If True, use the small vocabulary; otherwise, use the large vocabulary.

    Returns:
        str: The quality of the chord.
    """
    if use_small_vocab:
        # Small vocabulary (0-24)
        if id < 0 or id > 25:
            raise ValueError("Chord id must be in the range 0-24.")

        if id == 0:
            return "N"

        if id == 1:
            return "X"

        quality = "maj" if (id - 2) // 12 == 0 else "min"
        return quality

    else:
        # Large vocabulary (0-170)
        if id < 0 or id > 170:
            raise ValueError("Chord id must be in the range 0-170.")

        if id == 0:
            return "N"
        elif id == 1:
            return "X"

        # Subtract 2 to account for N and X
        id -= 2

        quality_index = id % 14

        templates = [
            "maj",
            "min",
            "dim",
            "aug",
            "min6",
            "maj6",
            "min7",
            "minmaj7",
            "maj7",
            "7",
            "dim7",
            "hdim7",
            "sus2",
            "sus4",
        ]

        quality = templates[quality_index]
        return quality


@lru_cache(maxsize=None)
def id_to_chord(chord_id: int, use_small_vocab: bool = SMALL_VOCABULARY) -> str:
    """
    Converts a chord id to a string representation of a chord.

    Args:
        chord_id (int): The chord id.
            - For small vocab: id in the range 0-24.
            - For large vocab: id in the range 0-170.
        use_small_vocab (bool): If True, use the small vocabulary; otherwise, use the large vocabulary.

    Returns:
        chord (str): The string representation of the chord.
    """
    if use_small_vocab:
        # Small vocabulary (0-24)
        if chord_id < 0 or chord_id > 25:
            raise ValueError("Chord id must be in the range 0-24.")

        if chord_id == 0:
            return "N"

        if chord_id == 1:
            return "X"

        root = (chord_id - 2) % 12
        quality = "maj" if (chord_id - 1) // 12 == 0 else "min"

        root_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        root = root_name[root]

        return f"{root}:{quality}"

    else:
        # Large vocabulary (0-170)
        if chord_id < 0 or chord_id > 170:
            raise ValueError("Chord id must be in the range 0-170.")

        if chord_id == 0:
            return "N"
        elif chord_id == 1:
            return "X"

        # Subtract 2 to account for N and X
        chord_id -= 2

        root = chord_id // 14
        quality_index = chord_id % 14

        root_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        root = root_name[root]

        templates = [
            "maj",
            "min",
            "dim",
            "aug",
            "min6",
            "maj6",
            "min7",
            "minmaj7",
            "maj7",
            "7",
            "dim7",
            "hdim7",
            "sus2",
            "sus4",
        ]

        quality = templates[quality_index]

        return f"{root}:{quality}"


@lru_cache(maxsize=None)
def transpose_chord_id(chord_id: int, semitones: int) -> int:
    """
    Apply a pitch shift to a list of chord ids.

    Args:
        chord_id (int): The chord id to shift.
        semitones (int): The number of semitones to shift the chord. In the range -5 to 6.

    Returns:
        chord_id_shifted (int): The shifted chord id.
    """

    # If the chord is N, return N
    if chord_id == 0:
        return 0

    if semitones == 0:
        return chord_id

    if semitones < -5 or semitones > 6:
        raise ValueError("Semitones must be in the range -5 to 6.")

    chord_id -= 2  # Offset by 2 for N and X
    chord_quality = chord_id // 12
    chord_root = chord_id % 12
    chord_root_shifted = (chord_root + semitones) % 12
    chord_id_shifted = chord_root_shifted + chord_quality * 12 + 2

    return chord_id_shifted


transpose_chord_id_vector = np.vectorize(transpose_chord_id)
id_to_chord_map = {i: id_to_chord(i) for i in range(NUM_CHORDS)}
id_to_chord_table = np.array([id_to_chord(i) for i in range(NUM_CHORDS)], dtype=object)
chord_to_id_map = {v: k for k, v in id_to_chord_map.items()}


def get_chord_annotation(
    filename: str,
    frame_length: float = 0.1,
    return_transitions: bool = False,
    override_dir: str = None,
) -> torch.Tensor:
    """
    Gets the chord annotation of an audio file as a tensor of chord ids.

    Args:
        filename (str): The filename of the audio file.
        frame_length (float): The length of each frame in seconds.

    Returns:
        chord_ids (torch.Tensor): A tensor of shape (num_frames,) where each element is a chord id.
    """

    chord_ann = get_raw_chord_annotation(filename, override_dir=override_dir)
    duration = chord_ann[-1].time + chord_ann[-1].duration

    # Loop over each frame and assign the chord
    num_frames = math.ceil(duration / frame_length)
    frames = torch.zeros(num_frames, dtype=torch.int64)
    current_chord_idx = 0
    previous_chord_id = None

    # If we want transitions, create a boolean tensor (False by default)
    if return_transitions:
        transitions = torch.zeros(num_frames, dtype=torch.bool)
    else:
        transitions = None
    for i in range(math.ceil(int(duration / frame_length))):
        time = i * frame_length + frame_length / 2
        while (
            chord_ann[current_chord_idx].time + chord_ann[current_chord_idx].duration
            < time
        ):
            current_chord_idx += 1
            # If we reach the end of the chord annotation, break
            if current_chord_idx >= len(chord_ann):
                current_chord_idx = len(chord_ann) - 1
                break

        chord_id = chord_to_id(chord_ann[current_chord_idx].value)
        frames[i] = chord_id

        if return_transitions:
            # We mark the first frame as a transition
            if i == 0:
                transitions[i] = True

            # If the chord changes, mark the frame as a transition
            if i != 0 and chord_id != previous_chord_id:
                boundary_time = chord_ann[current_chord_idx].time
                frame_start = time - (frame_length / 2)

                # Check if the chord boundary lies in this frame or the previous frame
                if frame_start < boundary_time:
                    transitions[i] = True
                else:
                    transitions[i - 1] = True

        previous_chord_id = chord_id

    if return_transitions:
        return frames, transitions
    return frames


def get_torch_device():
    """
    Get the torch device to use for training.

    Returns:
        torch.device: The torch device to use.
    """

    # CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")

    # MPS
    if torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS is available but not built.")
        return torch.device("mps")

    # CPU
    return torch.device("cpu")


def collate_fn(data: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for the DataLoader.

    Args:
        data (list): A list of tuples, where each tuple is a fixed length frame of features and chord annotation.

    Returns:
        cqt (torch.Tensor): The log CQT of the audio file. Has shape (num_frames, n_bins).
        chord_ids (torch.Tensor): A tensor of shape (num_frames,) where each element is a chord id.
    """
    cqt, chord_ids = zip(*data)
    # Stack the CQTs and chord annotations with padding if necessary
    cqt = torch.nn.utils.rnn.pad_sequence(cqt, batch_first=True, padding_value=0)
    chord_ids = torch.nn.utils.rnn.pad_sequence(
        chord_ids, batch_first=True, padding_value=-1
    )
    return cqt, chord_ids


def write_json(dictionary: dict, file: str):
    """
    Writes a dictionary to a JSON file.

    Args:
        json (dict): The dictionary to write to the JSON

    Returns:
        None
    """
    os.makedirs(os.path.dirname(file), exist_ok=True)  # Ensure directory exists
    with open(file, "w") as f:
        json.dump(dictionary, f)


def write_text(file: str, text: str):
    """
    Writes a string to a text file.

    Args:
        text (str): The string to write to the file
        file (str): The file to write to

    Returns:
        None
    """
    os.makedirs(os.path.dirname(file), exist_ok=True)  # Ensure directory exists
    with open(file, "w") as f:
        f.write(text)


def generate_experiment_name():
    """
    Generates a random experiment name.

    Returns:
        str: The experiment name.
    """
    adjectives = [
        "happy",
        "sad",
        "exciting",
        "boring",
        "fast",
        "slow",
        "loud",
        "quiet",
        "bright",
        "dark",
        "simple",
        "complex",
        "beautiful",
        "ugly",
        "calm",
        "chaotic",
        "melodic",
        "atonal",
        "jazzy",
        "classical",
    ]
    nouns = [
        "music",
        "song",
        "chord",
        "note",
        "rhythm",
        "harmony",
        "melody",
        "beat",
        "tune",
        "sound",
        "audio",
        "instrument",
        "genre",
        "tempo",
        "pitch",
        "timbre",
        "dynamics",
        "texture",
        "form",
        "structure",
    ]

    return f"{random.choice(adjectives)}-{random.choice(nouns)}-{random.randint(1000, 9999)}"


class EarlyStopper:
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float("inf")

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


if __name__ == "__main__":
    # Test get_pitch_classes
    print(get_chord_quality(13))
    print(get_chord_root(13))

    for i in range(2, 16):
        print(i, id_to_chord(i))
        print(get_chord_root(i), get_chord_quality(i))
