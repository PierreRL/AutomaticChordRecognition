"""
Script to compute beat-wise chord annotations (for chord loss) from raw chord annotations.
"""

import autorootcwd
import numpy as np
from typing import List
import torch


from src.utils import get_raw_beats, get_raw_chord_annotation, SR, HOP_LENGTH

def resample_boundaries(boundaries, beat_interval=1):
    """
    Resample beat boundaries based on the desired beat_interval.

    If beat_interval == 1, the original boundaries are returned.
    If beat_interval < 1, each interval is subdivided into int(round(1/beat_interval))
    equal parts.
    If beat_interval > 1, consecutive boundaries are grouped together by taking every
    grouping-th boundary.

    Parameters
    ----------
    original_boundaries : list of float
        Sorted list of boundary timestamps (in seconds), including the song start and end.
    beat_interval : float
        Desired beat interval rate. For example, 1 means one annotation per beat,
        0.5 inserts an annotation halfway through each beat (i.e. twice per beat),
        and 2 groups beats by 2.

    Returns
    -------
    boundaries : list of float
        The resampled list of boundaries.
    """
    if beat_interval == 1:
        return boundaries
    elif beat_interval < 1:
        subdivisions = int(round(1 / beat_interval))
        new_boundaries = []
        for i in range(len(boundaries) - 1):
            start_b = boundaries[i]
            end_b = boundaries[i + 1]
            # Generate boundaries within this interval.
            for j in range(subdivisions):
                new_boundaries.append(start_b + (end_b - start_b) * (j / subdivisions))
        # Ensure the final boundary is included.
        new_boundaries.append(boundaries[-1])
        return new_boundaries
    else:  # beat_interval > 1: group beats
        grouping = int(round(beat_interval))
        new_boundaries = boundaries[::grouping]
        if new_boundaries[-1] != boundaries[-1]:
            new_boundaries.append(boundaries[-1])
        return new_boundaries

def get_beatwise_chord_annotation(filename, beat_interval=1):
    """
    Compute beat-wise discrete chord annotations by aligning raw chord annotations
    with provided beat times.

    For each beat interval (between consecutive beat times), the chord is chosen
    as the one that overlaps the interval for the longest total duration.

    Parameters
    ----------
    filename : str
        The filename of the audio file (without extension) for which to compute
        beat-wise chord annotations.
    beat_interval : float
        The desired beat interval rate. For example, 1 means one annotation per beat,
        0.5 inserts an annotation halfway through each beat (i.e. twice per beat),
        and 2 groups beats by 2.

    Returns
    -------
    beat_chords : list of str
        List of chord labels (as strings), one per beat interval.
    """
    # Load chord annotations. Expected to be a sorted iterable of Observation objects.
    ann = get_raw_chord_annotation(filename)

    song_start = 0.0
    # Determine song end as the maximum end time among all chord observations.
    song_end = max(obs.time + obs.duration for obs in ann)

    beat_times = get_raw_beats(filename)

    # Filter out any beat times that occur after the song end.
    beat_times = [bt for bt in beat_times if bt <= song_end]

    # Create a full list of interval boundaries: from song_start, through the provided beat_times, to song_end.
    if beat_times[0] != 0:
        beat_times = [song_start] + list(beat_times)

    if beat_times[-1] < song_end:
        beat_times = list(beat_times) + [song_end]

    # Apply resampling of boundaries based on beat_interval.
    boundaries = resample_boundaries(boundaries, beat_interval)
    
    beat_chords = []

    for i in range(len(beat_times) - 1):
        beat_start = beat_times[i]
        beat_end = beat_times[i + 1]
        chord_overlaps = {}

        # For each chord observation, compute the overlap with the current beat interval.
        for obs in ann:
            obs_start = obs.time
            obs_end = obs.time + obs.duration
            # Calculate overlap duration between beat interval and chord segment.
            overlap = max(0.0, min(beat_end, obs_end) - max(beat_start, obs_start))
            if overlap > 0:
                chord_overlaps[obs.value] = chord_overlaps.get(obs.value, 0.0) + overlap

        # Select the chord with the maximum overlapping duration.
        selected_chord = max(chord_overlaps.items(), key=lambda x: x[1])[0]
        beat_chords.append(selected_chord)

    return beat_chords


def resample_features_by_beat(features, filename, beat_interval=1, hop_length=HOP_LENGTH, sample_rate=SR):
    """
    Resample a feature matrix to beat-synchronous representation by aggregating (averaging)
    the frames that fall into each beat interval. This implementation assumes that
    features is a torch tensor.

    Parameters:
      features    : torch.Tensor of shape (n_frames, feature_dim)
                    The feature matrix (e.g., CQT), where each row corresponds to a frame.
      filename    : str
                    Filename used to retrieve raw beat times and chord annotations.
      beat_interval : float, optional (default=1)
                    The desired beat interval rate. For example, 1 means one annotation per beat;
                    0.5 inserts an annotation halfway through each beat.
      hop_length  : int
                    The hop length (in samples) used when computing the features.
      sample_rate : int, optional (default=SR)
                    The sample rate (in Hz) of the audio from which features were computed.

    Returns:
      beat_features : torch.Tensor of shape (n_beats, feature_dim)
                      A new feature matrix where each row is the aggregated (averaged)
                      feature vector corresponding to a beat interval.
    """
    # Get raw beats and chord annotations.
    beat_times = get_raw_beats(filename)
    ann = get_raw_chord_annotation(filename)
    song_start = 0.0
    # Determine song end as the maximum end time among all chord observations.
    song_end = max(obs.time + obs.duration for obs in ann)
    beat_times = [bt for bt in beat_times if bt <= song_end]
    # Ensure boundaries include song start and song end.
    if beat_times[0] != 0:
        beat_times = [song_start] + List(beat_times)
    if beat_times[-1] < song_end:
        beat_times = List(beat_times) + [song_end]

    # Resample beat times based on the desired beat_interval.
    beat_times = resample_boundaries(beat_times, beat_interval)
    
    # Compute the time for each frame using the hop_length and sample_rate.
    n_frames = features.shape[0]
    frame_times = (torch.arange(n_frames, device=features.device, dtype=torch.float) * hop_length + hop_length / 2) / sample_rate

    beat_features = []  # List to collect beat-level features

    # Iterate over beat intervals.
    for i in range(len(beat_times) - 1):
        start_time = beat_times[i]
        end_time = beat_times[i + 1]
        
        # Find the indices of frames that fall inside the current beat interval.
        indices = torch.where((frame_times >= start_time) & (frame_times < end_time))[0]
        
        if indices.numel() == 0:
            raise ValueError(f"No frames found for beat interval: {start_time} - {end_time}")

        # Average the features of all frames in this interval.
        beat_vec = torch.mean(features[indices], dim=0)
        beat_features.append(beat_vec)

    # Optionally, process the final beat: average all frames starting at the last beat time.
    last_start = beat_times[-1]
    indices = torch.where(frame_times >= last_start)[0]
    if indices.numel() > 0:
        beat_features.append(torch.mean(features[indices], dim=0))

    return torch.stack(beat_features, dim=0)