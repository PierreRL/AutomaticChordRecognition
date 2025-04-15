"""
Script to compute beat-wise chord annotations (for chord loss) from raw chord annotations.
"""

import autorootcwd
import torch

from src.utils import (
    get_raw_beats,
    get_raw_chord_annotation,
    chord_to_id,
    SR,
    HOP_LENGTH,
)


def get_resampled_full_beats(
    filename: str, 
    beat_interval: int = 1, 
    perfect_beat_resample: bool = False, 
    override_dir_chord: str = None, 
    override_dir_beat: str = None,
    song_end:float = None
) -> list:
    """
    Compute beat-wise chord annotations by aligning raw chord annotations with provided beat times.

    For each beat interval (between consecutive beat times), the chord is chosen
    as the one that overlaps the interval for the longest total duration.

    Returns 'full' beats in that it includes 0 and the song end time.

    Parameters
    ----------
    filename : str
        The filename of the audio file (without extension) for which to compute
        beat-wise chord annotations.
    beat_interval : float
        The desired beat interval rate. For example, 1 means one annotation per beat,
        0.5 inserts an annotation halfway through each beat (i.e. twice per beat),
        and 2 groups beats by 2.
    perfect_beat_resample : bool
        If True, use perfect beat times from the chord annotations instead of the raw beat times.
    override_dir_chord : str
        Directory to override the default chord annotation directory.
    override_dir_beat : str
        Directory to override the default beat annotation directory.
    song_end : float
        The end time of the song. If None, it will be computed from the chord annotations.

    Returns
    -------
    beat_chords : list of str
        List of chord labels (as strings), one per beat interval.
    """
    # Load chord annotations. Expected to be a sorted iterable of Observation objects.
    ann = get_raw_chord_annotation(filename, override_dir_chord)
    if perfect_beat_resample:
        beat_times = get_perfect_beats_from_ann(ann)
    else:
        beat_times = get_raw_beats(filename, override_dir_beat)

    # Filter out any beat times that occur after the song end.
    song_start = 0.0
    song_end = min(max(obs.time + obs.duration for obs in ann), song_end)
    beat_times = [bt for bt in beat_times if bt <= song_end]

    # Remove duplicates and sort the list.
    beat_times = sorted(set(beat_times))

    # Remove first and last beat time if within 0.25 seconds of song start or end.
    if beat_times[0] < song_start + 0.25:
        beat_times = beat_times[1:]
    if beat_times[-1] > song_end - 0.25:
        beat_times = beat_times[:-1]

    # Create a full list of interval boundaries: from song_start, through the provided beat_times, to song_end.
    if beat_times[0] != 0:
        beat_times = [song_start] + list(beat_times)

    if beat_times[-1] < song_end:
        beat_times = list(beat_times) + [song_end]

    return resample_boundaries(beat_times, beat_interval)


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
        # Assert that beat_interval is a positive float that is 1 / n for some n.
        assert (
            0 < beat_interval < 1
        ), "beat_interval must be a positive float less than 1"
        assert 1 / beat_interval == int(
            1 / beat_interval
        ), "beat_interval must be 1 / n for some n"
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
        assert (
            beat_interval.is_integer()
        ), "beat_interval must be an integer when greater than 1"
        grouping = int(round(beat_interval))
        new_boundaries = boundaries[::grouping]
        if new_boundaries[-1] != boundaries[-1]:
            new_boundaries.append(boundaries[-1])
        return new_boundaries


def get_perfect_beats_from_ann(ann):
    """
    Compute perfect beat times from the raw chord annotations.

    Parameters
    ----------
    ann : list of Observation
        List of chord annotations.

    Returns
    -------
    perfect_beats : list of float
        List of perfect beat times.
    """
    perfect_beats = []
    for obs in ann:
        # Add the start and end times to the perfect beats.
        perfect_beats.append(obs.time)
    perfect_beats = perfect_beats[1:]  # Remove the first beat time
    # Remove duplicates and sort the list.
    return sorted(set(perfect_beats))


def get_beatwise_chord_annotation(
    filename, beat_interval=1, 
    perfect_beat_resample=False, 
    return_as_string=False, 
    override_dir_chord=None, 
    override_dir_beat=None,
    song_end=None
):
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
    perfect_beat_resample : bool
        If True, use perfect beat times from the chord annotations instead of the raw beat times.
    return_as_string : bool
        If True, return the chord annotations as a list of strings instead of a tensor of chord ids.
    override_dir_chord : str
        Directory to override the default chord annotation directory.
    override_dir_beat : str
        Directory to override the default beat annotation directory.
    song_end : float
        The end time of the song. If None, it will be computed from the chord annotations.
    Returns
    -------
    beat_chords : list of str or torch.Tensor
    """
    # Load chord annotations. Expected to be a sorted iterable of Observation objects.
    beat_times = get_resampled_full_beats(
        filename, 
        beat_interval, 
        perfect_beat_resample, 
        override_dir_chord=override_dir_chord, 
        override_dir_beat=override_dir_beat,
        song_end=song_end
    )
    ann = get_raw_chord_annotation(filename, override_dir_chord)

    # Adjust the song end time if provided.
    song_end = min(max(obs.time + obs.duration for obs in ann), song_end)

    beat_chords = []

    for i in range(len(beat_times) - 1):
        beat_start = beat_times[i]
        beat_end = beat_times[i + 1]

        if beat_start >= song_end:
            break
        if beat_start >= ann[-1].time + ann[-1].duration:
            break
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

    if return_as_string:
        return beat_chords

    return torch.tensor([chord_to_id(chord) for chord in beat_chords])


def resample_features_by_beat(
    features,
    filename,
    beat_interval=1,
    frame_rate=SR / HOP_LENGTH,
    perfect_beat_resample=False,
    override_dir_chord=None,
    override_dir_beat=None,
    song_end=None,
):
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
      frame_rate  : float, optional (default=100.0)
                    The frame rate (in Hz), i.e., number of frames per second.
        perfect_beat_resample : bool, optional (default=False)  
                    If True, use perfect beat times from the chord annotations instead of the raw beat times.
        override_dir_chord : str, optional
                    Directory to override the default chord annotation directory.
        override_dir_beat : str, optional
                    Directory to override the default beat annotation directory.
        song_end : float, optional
                    The end time of the song. If None, it will be computed from the chord annotations.

    Returns:
      beat_features : torch.Tensor of shape (n_beats, feature_dim)
                      A new feature matrix where each row is the aggregated (averaged)
                      feature vector corresponding to a beat interval.
    """

    beat_times = get_resampled_full_beats(
        filename, beat_interval, perfect_beat_resample, override_dir_chord=override_dir_chord, override_dir_beat=override_dir_beat, song_end=song_end
    )

    # Calculate frame times using the frame rate
    n_frames = features.shape[0]
    frame_times = (
        torch.arange(n_frames, device=features.device, dtype=torch.float) + 0.5
    ) / frame_rate

    beat_features = []

    for i in range(len(beat_times) - 1):
        start_time = beat_times[i]
        end_time = beat_times[i + 1]
        indices = torch.where((frame_times >= start_time) & (frame_times < end_time))[0]

        if indices.numel() == 0:
            raise ValueError(
                f"No frames found for beat interval: {start_time} - {end_time}"
            )

        beat_vec = torch.mean(features[indices], dim=0)
        beat_features.append(beat_vec)

    # Optionally average frames after the last beat time
    last_start = beat_times[-1]
    indices = torch.where(frame_times >= last_start)[0]
    if indices.numel() > 0:
        beat_features.append(torch.mean(features[indices], dim=0))

    return torch.stack(beat_features, dim=0)
