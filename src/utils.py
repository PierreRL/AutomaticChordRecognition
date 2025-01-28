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

# Music processing libraries
import torch_pitch_shift
import librosa
import jams
from harte.harte import Harte

SMALL_VOCABULARY = False

if SMALL_VOCABULARY:
    NUM_CHORDS = 25
else:
    NUM_CHORDS = 170


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


@lru_cache(maxsize=None)
def get_raw_chord_annotation(filename):
    """
    Retrieves the raw chord annotation data from a JAMS file.

    Args:
        filename (str): The filename of the JAMS file to load.

    Returns:
        chord_annotation (SortedKeyDict): An ordered dict of chord annotations.
    """
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
    sr: int = 44100,
    hop_length: int = 4096,
    n_bins: int = 36 * 6,
    bins_per_octave: int = 36,
    fmin: float = librosa.note_to_hz("C1"),
    absolute: bool = True,
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

    return torch.tensor(cqt, dtype=torch.float32).T


def pitch_shift_cqt(
    cqt: torch.Tensor, semitones: int, bins_per_octave=36
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
    sr: int = 44100,
    hop_length: int = 4096,
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
        return 1 if not use_small_vocab else 0

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
            return 0  # Map unknown chords to N

        return root + 12 * quality + 1

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
        if id < 0 or id > 24:
            raise ValueError("Chord id must be in the range 0-24.")

        if id == 0:
            return "N"

        root_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        root = root_name[(id - 1) % 12]

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
        if id < 0 or id > 24:
            raise ValueError("Chord id must be in the range 0-24.")

        if id == 0:
            return "N"

        quality = "maj" if (id - 1) // 12 == 0 else "min"
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
def id_to_chord(chord_id: int, use_small_vocab: bool = False) -> str:
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
        if chord_id < 0 or chord_id > 24:
            raise ValueError("Chord id must be in the range 0-24.")

        if chord_id == 0:
            return "N"

        root = (chord_id - 1) % 12
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

    chord_quality = chord_id // 12
    chord_root = chord_id % 12
    chord_root_shifted = (chord_root + semitones) % 12
    chord_id_shifted = chord_root_shifted + chord_quality * 12

    return chord_id_shifted


transpose_chord_id_vector = np.vectorize(transpose_chord_id)
id_to_chord_map = {i: id_to_chord(i) for i in range(NUM_CHORDS)}
id_to_chord_table = np.array([id_to_chord(i) for i in range(NUM_CHORDS)], dtype=object)
chord_to_id_map = {v: k for k, v in id_to_chord_map.items()}


def get_chord_annotation(
    filename: str,
    frame_length: float = 0.1,
    return_transitions: bool = False,
) -> torch.Tensor:
    """
    Gets the chord annotation of an audio file as a tensor of chord ids.

    Args:
        filename (str): The filename of the audio file.
        frame_length (float): The length of each frame in seconds.

    Returns:
        chord_ids (torch.Tensor): A tensor of shape (num_frames,) where each element is a chord id.
    """

    chord_ann = get_raw_chord_annotation(filename)
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


def collate_fn(data: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
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
    with open(file, "w") as f:
        json.dump(dictionary, f)


def write_text(text: str, file: str):
    """
    Writes a string to a text file.

    Args:
        text (str): The string to write to the file
        file (str): The file to write to

    Returns:
        None
    """
    with open(file, "w") as f:
        f.write(text)


if __name__ == "__main__":
    # Test get_pitch_classes
    print(get_chord_quality(13))
    print(get_chord_root(13))

    for i in range(2, 16):
        print(i, id_to_chord(i))
        print(get_chord_root(i), get_chord_quality(i))
