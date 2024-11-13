"""
Various utility functions for processing music data.
"""

import os
import math
import numpy as np
import torch

# Music processing libraries
import librosa
import jams
from harte.harte import Harte


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


def get_chord_annotation(filename):
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
    sr: int = 22050,
    hop_length: int = 2048,
    n_bins: int = 24 * 6,
    bins_per_octave: int = 24,
    fmin: float = librosa.note_to_hz("C1"),
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
    # Default hyperparameters from https://arxiv.org/pdf/1907.02698.pdf
    src = librosa.load(
        os.path.join("./data/processed/audio/", f"{filename}.mp3"), sr=sr
    )[0]
    cqt = librosa.amplitude_to_db(
        np.abs(
            librosa.cqt(
                src,
                sr=sr,
                hop_length=hop_length,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                fmin=fmin,
            )
        )
    )
    return torch.tensor(cqt, dtype=torch.float32).T


def chord_to_id(chord: str) -> torch.Tensor:
    """
    Convert a chord to an index corresponding to a chord id.

    Args:
        chord (str): Harte string representation of a chord.

    Returns:
        torch.Tensor: The chord id in the range 0-24. 0 is reserved for N (no chord).
    """
    # If the chord is N, return 0
    if chord == "N" or chord == "X":
        return 0

    # Convert the chord to a Harte object
    chord = Harte(chord)

    # Get the root of the chord as a pitch class (0-11)
    root = chord.root().pitchClass

    # Major is 0, minor is 1
    quality = chord.quality
    if quality == "major":
        quality = 0
    elif quality == "minor":
        quality = 1
    else:
        return 0

    # Return the chord id in the range 1-24
    return root + 12 * quality + 1


def id_to_chord(chord_id: int) -> str:
    """
    Converts a chord id to a string representation of a chord.

    Args:
        chord_id (int): The chord id in the range 0-24.

    Returns:
        chord (str): The string representation of the chord.
    """

    if chord_id < 0 or chord_id > 24:
        raise ValueError("Chord id must be in the range 0-24.")

    # If the chord is N, return N
    if chord_id == 0:
        return "N"

    # Convert the chord id to a Harte object
    root = chord_id % 12
    quality = "maj" if chord_id // 12 == 0 else "min"

    # Convert the root to a string
    root_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    root = root_name[root]

    # Return the string representation of the chord
    return f"{root}:{quality}"


def chord_ann_to_tensor(
    filename: str,
    frame_length: float = 0.1,
) -> torch.Tensor:
    """
    Convert a chord annotation to a tensor of chord ids.

    Args:
        filename (str): The filename of the audio file.
        frame_length (float): The length of each frame in seconds.

    Returns:
        chord_ids (torch.Tensor): A tensor of shape (num_frames,) where each element is a chord id.
    """

    chord_ann = get_chord_annotation(filename)
    duration = chord_ann[-1].time + chord_ann[-1].duration

    # Loop over each frame and assign the chord
    frames = torch.zeros(math.ceil(duration / frame_length), dtype=torch.int64)
    current_chord_idx = 0
    for i in range(math.ceil(int(duration / frame_length))):
        time = i * frame_length + frame_length / 2
        while (
            chord_ann[current_chord_idx].time + chord_ann[current_chord_idx].duration
            < time
        ):
            current_chord_idx += 1

        chord_id = chord_to_id(chord_ann[current_chord_idx].value)
        frames[i] = chord_id

    return frames
