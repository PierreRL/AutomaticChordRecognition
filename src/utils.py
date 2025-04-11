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
import re
import unicodedata
from functools import lru_cache
from typing import List, Tuple, Union

# Music processing libraries
import librosa
import jams
from harte.harte import Harte

# Constants
SMALL_VOCABULARY = False
SR = 44100
HOP_LENGTH = 4096
BINS_PER_OCTAVE = 36
N_BINS = BINS_PER_OCTAVE * 6
N_MELS = BINS_PER_OCTAVE * 6
N_FFT = 2048

if SMALL_VOCABULARY:
    NUM_CHORDS = 26
else:
    NUM_CHORDS = 170


# Functions
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


def get_split_filenames(dir: str = "data/processed/") -> Tuple[List, List, List]:
    """Get the filenames for the train, validation, and test sets.

    Returns:

        train_filenames (list): The filenames for the training set.
        val_filenames (list): The filenames for the validation set.
        test_filenames (list): The filenames for the test set."""
    with open(f"{dir}/splits.json", "r") as f:
        splits = json.load(f)

    return splits["train"], splits["val"], splits["test"]


def get_raw_beats(filename, override_dir=None):
    """
    Retrieves the raw beat annotation pre-computed using madmom.

    Args:
        filenames (list): The filenames of the JAMS files to load.

    Returns:
        beats (ndarray): A list of beats for each file.
    """
    filename = sanitize_filename(filename)
    if override_dir is not None:
        return np.load(os.path.join(f"{override_dir}/", f"{filename}.npy"))

    return np.load(os.path.join("./data/processed/beats/", f"{filename}.npy"))


def sanitize_filename(name):
    # Normalize Unicode (e.g., é vs e + ́)
    name = unicodedata.normalize("NFC", name)

    # Lowercase
    name = name.lower()

    # Replace spaces with underscores
    name = name.replace(" ", "_")

    # Remove all characters *except* letters, numbers, underscores, hyphens, and dots
    name = re.sub(r"[^\w\-.]", "", name)

    return name


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
        jam = jams.load(os.path.join(f"{override_dir}/", f"{filename}.jams"))
    else:
        jam = jams.load(os.path.join("./data/processed/chords/", f"{filename}.jams"))
    chord_ann = jam.annotations.search(namespace="chord")[0]
    return chord_ann.data


def get_chord_seq(
    filename, override_dir: str, use_small_vocab: bool = SMALL_VOCABULARY
):
    """
    Retrieves the chord sequence from a JAMS file and the associated 'beats'.

    Args:
        filename (str): The filename of the JAMS file to load.

    Returns:
        chord_seq (list[str]): A list of chord annotations.
    """
    chord_ann = get_raw_chord_annotation(filename, override_dir=override_dir)
    seq = [chord.value for chord in chord_ann]

    if use_small_vocab:
        seq = [chord_to_id(chord, use_small_vocab=True) for chord in seq]
        seq = [id_to_chord(chord, use_small_vocab=True) for chord in seq]

    beats = [obs.time for obs in chord_ann]
    beats = beats + [chord_ann[-1].time + chord_ann[-1].duration]  # Add the last 'beat'
    return seq, beats


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


def load_audio(filename: str, override_dir: str = None, sr: int = SR) -> np.ndarray:
    path = os.path.join(override_dir or "./data/processed/audio/", f"{filename}.mp3")
    return librosa.load(path, sr=sr)[0]


def to_tensor_if_needed(
    x: np.ndarray, return_as_tensor: bool = True
) -> Union[torch.Tensor, np.ndarray]:
    x = x.T  # (freq, time) → (time, freq)
    return torch.tensor(x, dtype=torch.float32) if return_as_tensor else x


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
):
    src = load_audio(filename, override_dir, sr)
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
    return to_tensor_if_needed(cqt, return_as_tensor)


def get_mel_spectrogram(
    filename: str,
    override_dir: str = None,
    sr: int = SR,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    fmin: float = librosa.note_to_hz("C1"),
    absolute: bool = True,
    return_as_tensor: bool = True,
):
    src = load_audio(filename, override_dir, sr)
    mel = librosa.feature.melspectrogram(
        y=src, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin
    )
    if absolute:
        mel = np.abs(mel)
    mel = librosa.amplitude_to_db(mel)
    return to_tensor_if_needed(mel, return_as_tensor)


def get_chroma_cqt(
    filename: str,
    override_dir: str = None,
    sr: int = SR,
    hop_length: int = HOP_LENGTH,
    bins_per_octave: int = BINS_PER_OCTAVE,
    fmin: float = librosa.note_to_hz("C1"),
    return_as_tensor: bool = True,
):
    src = load_audio(filename, override_dir, sr)
    chroma = librosa.feature.chroma_cqt(
        y=src,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
        n_chroma=12,
        n_octaves=6,
    )
    return to_tensor_if_needed(chroma, return_as_tensor)


def get_linear_spectrogram(
    filename: str,
    override_dir: str = None,
    sr: int = SR,
    hop_length: int = HOP_LENGTH,
    n_fft: int = N_FFT,
    absolute: bool = True,
    return_as_tensor: bool = True,
):
    src = load_audio(filename, override_dir, sr)
    spec = librosa.stft(src, n_fft=n_fft, hop_length=hop_length, window="hann")
    if absolute:
        spec = np.abs(spec)
    spec = librosa.amplitude_to_db(spec)
    return to_tensor_if_needed(spec, return_as_tensor)


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
def get_pitch_classes_from_id(
    chord_id: int, use_small_vocab: bool = SMALL_VOCABULARY
) -> torch.Tensor:
    """
    Get the pitch classes from a chord id. These are not root transposed.

    Args:
        chord_id (int): The chord id.

    Returns:
        Tensor (long): A binary vector of length 12, where each index corresponds to a pitch class.
    """
    if use_small_vocab:
        raise NotImplementedError("Small vocabulary get pitch classes not implemented")

    # If the chord id is out of range, raise an error
    if chord_id < 0 or chord_id > 170:
        raise ValueError("Chord id must be in the range 0-170.")
    # If the chord is N or X, return a Tensor of zeros 12 long
    if chord_id == 0 or chord_id == 1:
        return torch.zeros(12, dtype=torch.long)

    # Get quality and root from the chord id
    root = get_chord_root(chord_id, return_idx=True) - 2  # Offset by 2 for N and X
    quality_idx = get_chord_quality(chord_id, return_idx=True)

    # Define the templates
    templates = [
        [0, 4, 7],  # Major
        [0, 3, 7],  # Minor
        [0, 3, 6],  # Diminished
        [0, 4, 8],  # Augmented
        [0, 3, 7, 9],  # Minor 6
        [0, 4, 7, 9],  # Major 6
        [0, 3, 7, 10],  # Minor 7
        [0, 3, 7, 11],  # Minor Major 7
        [0, 4, 7, 11],  # Major 7
        [0, 4, 7, 10],  # 7
        [0, 3, 6, 9],  # Diminished 7
        [0, 3, 6, 10],  # Half Diminished 7
        [0, 2, 7],  # Suspended 2
        [0, 5, 7],  # Suspended 4
    ]
    # Get the quality from the templates
    pitches_quality = templates[quality_idx]

    # Add root to each pitch class
    pitch_classes = [(pitch + root) % 12 for pitch in pitches_quality]

    # Create a binary vector of length 12
    binary_vector = torch.zeros(12, dtype=torch.long)

    # Set the corresponding indices to 1 for the pitch classes
    for pitch in pitch_classes:
        binary_vector[pitch] = 1  # Ensure pitch is in the range 0-11

    return binary_vector


def get_pitch_classes_batch(
    chord_ids: torch.Tensor, use_small_vocab: bool = False
) -> torch.Tensor:
    """
    Get the pitch classes for a batch of chord IDs, applied over a sequence.

    Args:
        chord_ids (Tensor): A tensor of shape (B) where each element is a chord_id.
        use_small_vocab (bool): Whether to use a small vocabulary (default: False).

    Returns:
        Tensor: A tensor of binary vectors (batch_size, T, 12) representing the pitch classes.
    """

    # Apply the get_pitch_classes_from_id function to each element in the flattened tensor
    pitch_classes = torch.stack(
        [
            get_pitch_classes_from_id(chord_id.item(), use_small_vocab)
            for chord_id in chord_ids
        ]
    )
    return pitch_classes


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
def get_chord_root(
    id: int, use_small_vocab: bool = SMALL_VOCABULARY, return_idx: bool = False
) -> str:
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
            return "N" if not return_idx else 0

        if id == 1:
            return "X" if not return_idx else 1

        idx = (id - 2) % 12
        if return_idx:
            return idx + 2  # Offset by 2 for N and X

        root_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        root = root_name[idx]

        return root

    else:
        # Large vocabulary (0-169)
        if id < 0 or id > 169:
            raise ValueError(f"Chord id must be in the range 0-170, but got {id}.")

        if id == 0:
            return "N" if not return_idx else 0
        elif id == 1:
            return "X" if not return_idx else 1

        # Subtract 2 to account for N and X
        id -= 2
        root = id // 14
        if return_idx:
            return root + 2  # Offset by 2 for N and X

        root_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        root = root_name[root]

        return root


def get_chord_root_batch(chord_ids: torch.Tensor) -> torch.Tensor:
    """
    Get the root of the chord for a batch of chord IDs, applied over a sequence.

    Args:
        chord_ids (Tensor): A tensor of shape (batch_size, T) where each element is a chord_id.
        return_as_idx (bool): Whether to return the root as an index (default: False).

    Returns:
        Tensor: A tensor of roots with shape (batch_size, T).
    """
    # Flatten the input tensor (batch_size * T,)
    flat_chord_ids = chord_ids.view(-1)

    # Apply the get_chord_root function to each element in the flattened tensor
    roots = (
        torch.Tensor(
            [
                get_chord_root(chord_id.item(), return_idx=True)
                for chord_id in flat_chord_ids
            ]
        )
        .view(chord_ids.size())
        .long()
    )

    return roots


@lru_cache(maxsize=None)
def get_chord_quality(
    id: int, use_small_vocab: bool = SMALL_VOCABULARY, return_idx: bool = False
) -> str:
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
            return "N" if not return_idx else 0

        if id == 1:
            return "X" if not return_idx else 1

        idx = (id - 2) // 12
        if return_idx:
            return idx

        quality = "maj" if idx == 0 else "min"
        return quality

    else:
        # Large vocabulary (0-170)
        if id < 0 or id > 170:
            raise ValueError("Chord id must be in the range 0-170.")

        if id == 0:
            return "N" if not return_idx else 0
        elif id == 1:
            return "X" if not return_idx else 1

        # Subtract 2 to account for N and X
        id -= 2

        quality_index = id % 14

        if return_idx:
            return quality_index

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
        quality = "maj" if (chord_id - 2) // 12 == 0 else "min"

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
        root = (chord_id - 2) // 14
        quality_index = (chord_id - 2) % 14

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

    # If the chord is X, return X
    if chord_id == 1:
        return 1

    if semitones == 0:
        return chord_id

    if semitones < -5 or semitones > 6:
        raise ValueError("Semitones must be in the range -5 to 6.")

    # chord_id -= 2  # Offset by 2 for N and X
    chord_quality = (chord_id - 2) % 14
    chord_root = (chord_id - 2) // 14
    chord_root_shifted = (chord_root + semitones) % 12
    chord_id_shifted = (chord_root_shifted * 14) + chord_quality + 2

    return chord_id_shifted


transpose_chord_id_vector = np.vectorize(transpose_chord_id)
id_to_chord_map = {i: id_to_chord(i) for i in range(NUM_CHORDS)}
id_to_chord_table = np.array([id_to_chord(i) for i in range(NUM_CHORDS)], dtype=object)
chord_to_id_map = {v: k for k, v in id_to_chord_map.items()}


def get_chord_annotation(
    filename: str,
    frame_length: float = HOP_LENGTH / SR,
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
        transitions (torch.Tensor): A tensor of shape (num_frames,) where each element is True if the chord changes in that frame.
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


def collate_fn(data: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for the DataLoader.

    Args:
        data (list): A list of tuples, where each tuple is a fixed length frame of features and chord annotation.

    Returns:
        cqt (torch.Tensor): The log CQT of the audio file. Has shape (num_frames, n_bins).
        chord_ids (torch.Tensor): A tensor of shape (num_frames,) where each element is a chord id.
    """
    cqt, gens, chord_ids = zip(*data)
    # Stack the CQTs and chord annotations with padding if necessary
    cqt = torch.nn.utils.rnn.pad_sequence(cqt, batch_first=True, padding_value=0)
    gens = torch.nn.utils.rnn.pad_sequence(gens, batch_first=True, padding_value=0)
    chord_ids = torch.nn.utils.rnn.pad_sequence(
        chord_ids, batch_first=True, padding_value=-1
    )
    return cqt, gens, chord_ids


def collate_fn_indexed(
    data: List[Tuple],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], List[str]]:
    """
    Collate function for the DataLoader with filenames.

    Args:
        data (list): A list of tuples, where each tuple is a fixed length frame of features and chord annotation.

    Returns:
        cqt (torch.Tensor): The log CQT of the audio file. Has shape (num_frames, n_bins).
        chord_ids (torch.Tensor): A tensor of shape (num_frames,) where each element is a chord id.
        filenames (list): A list of filenames corresponding to the data.
    """
    data_items, idxs = zip(*data)
    cqt, gens, chord_ids = zip(*data_items)
    # Stack the CQTs and chord annotations with padding if necessary
    cqt = torch.nn.utils.rnn.pad_sequence(cqt, batch_first=True, padding_value=0)
    gens = torch.nn.utils.rnn.pad_sequence(gens, batch_first=True, padding_value=0)
    chord_ids = torch.nn.utils.rnn.pad_sequence(
        chord_ids, batch_first=True, padding_value=-1
    )
    idxs = list(idxs)
    return (cqt, gens, chord_ids), idxs


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
    # Test get_pitch_classes_from_id
    chord_id = chord_to_id("G#:dim7")
    # Get pitch classes
    pitch_classes = get_pitch_classes_from_id(chord_id)
    root = (get_chord_root(chord_id, return_idx=True) - 2) % 12
    print(f"Chord id: {chord_id}")
    print(f"Pitch classes: {pitch_classes}")
    print(f"Root: {root}")
