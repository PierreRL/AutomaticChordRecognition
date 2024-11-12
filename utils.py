import os
import jams
import librosa
import numpy as np


def get_chord_annotation(filename):
    jam = jams.load(os.path.join("./data/processed/chords/", f"{filename}.jams"))
    chord_ann = jam.annotations.search(namespace="chord")[0]
    return chord_ann.data


def get_annotation_metadata(filename):
    jam = jams.load(os.path.join("./data/processed/chords/", f"{filename}.jams"))
    metadata = jam.file_metadata
    return metadata


def get_log_cqt(
    filename,
    sr=22050,
    hop_length=2048,
    n_bins=24 * 6,
    bins_per_octave=24,
    fmin=librosa.note_to_hz("C1"),
):
    # Default hyperparameters from https://arxiv.org/pdf/1907.02698.pdf
    src = librosa.load(
        os.path.join("./data/processed/audio/", f"{filename}.mp3"), sr=sr
    )[0]
    log_cqt = librosa.amplitude_to_db(
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
    return log_cqt
