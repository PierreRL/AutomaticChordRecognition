import autorootcwd
import random

ROOTS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

################################################################################
# Weighted chord pools: each item is (chord_quality, probability).
# Adjust probabilities as you like. They don't have to sum to exactly 1,
# but it's conceptually tidy if they do.
################################################################################

MAJOR_CHORD_POOLS = {
    # I
    1: [
        ("maj", 0.2),
        ("maj7", 0.4),
        ("maj6", 0.2),
        ("sus4", 0.2)
    ],
    # ii
    2: [
        ("min", 0.2),
        ("min7", 0.4),
        ("dim", 0.1),
        ("hdim7", 0.1),
        ("sus2", 0.2)
    ],
    # iii
    3: [
        ("min", 0.3),
        ("min7", 0.4),
        ("min6", 0.2),
        ("dim", 0.1)
    ],
    # IV
    4: [
        ("maj", 0.3),
        ("maj7", 0.4),
        ("maj6", 0.2),
        ("sus4", 0.1)
    ],
    # V
    5: [
        ("maj", 0.1),
        ("7", 0.6),
        ("aug", 0.1),
        ("sus4", 0.1),
        ("dim7", 0.1)
    ],
    # vi
    6: [
        ("min", 0.2),
        ("min7", 0.6),
        ("min6", 0.2)
    ],
    # vii
    7: [
        ("dim", 0.3),
        ("dim7", 0.4),
        ("hdim7", 0.3)
    ]
}

MINOR_CHORD_POOLS = {
    # i
    1: [
        ("min", 0.3),
        ("min7", 0.4),
        ("min6", 0.2),
        ("minmaj7", 0.1)
    ],
    # ii° (or half-diminished)
    2: [
        ("dim", 0.3),
        ("hdim7", 0.4),
        ("sus2", 0.3)
    ],
    # III
    3: [
        ("maj", 0.4),
        ("maj7", 0.4),
        ("maj6", 0.2)
    ],
    # iv
    4: [
        ("min", 0.4),
        ("min7", 0.3),
        ("dim", 0.3)
    ],
    # V (harmonic minor's V is often a dominant 7 or aug)
    5: [
        ("7", 0.5),
        ("aug", 0.3),
        ("dim7", 0.2)
    ],
    # VI
    6: [
        ("maj", 0.4),
        ("maj7", 0.4),
        ("maj6", 0.2)
    ],
    # VII
    7: [
        ("7", 0.3),
        ("dim7", 0.3),
        ("hdim7", 0.4)
    ]
}

################################################################################
# 3) Scale-building & chord-degree identification
################################################################################

def build_scale(key_root, minor=False):
    MAJOR_SCALES = {
        "C":  ["C",  "D",  "E",  "F",  "G",  "A",  "B"],
        "Db": ["Db", "Eb", "F",  "Gb", "Ab", "Bb", "C"],
        "D":  ["D",  "E",  "F#", "G",  "A",  "B",  "C#"],
        "Eb": ["Eb", "F",  "G",  "Ab", "Bb", "C",  "D"],
        "E":  ["E",  "F#", "G#", "A",  "B",  "C#", "D#"],
        "F":  ["F",  "G",  "A",  "Bb", "C",  "D",  "E"],
        "Gb": ["Gb", "Ab", "Bb", "Cb", "Db", "Eb", "F"],
        "G":  ["G",  "A",  "B",  "C",  "D",  "E",  "F#"],
        "Ab": ["Ab", "Bb", "C",  "Db", "Eb", "F",  "G"],
        "A":  ["A",  "B",  "C#", "D",  "E",  "F#", "G#"],
        "Bb": ["Bb", "C",  "D",  "Eb", "F",  "G",  "A"],
        "B":  ["B",  "C#", "D#", "E",  "F#", "G#", "A#"]
    }
    scale = MAJOR_SCALES[key_root]
    if minor:
        # Natural minor = rotate scale to start at 6th degree
        scale = scale[5:] + scale[:5]
    return scale

def get_scale_degree(chord, scale):
    """Given a chord in Harte notation (e.g. 'C:maj7'), parse out the root and see which scale degree it is."""
    if ":" in chord:
        root = chord.split(":", 1)[0]
    else:
        root = chord
    if root in scale:
        return scale.index(root) + 1
    return None

################################################################################
# 4) Weighted random choice from chord pools
################################################################################

def pick_chord_quality(chord_pool):
    """Given a list of (quality, weight), pick one by weighted probability."""
    qualities = [q for (q, w) in chord_pool]
    weights = [w for (q, w) in chord_pool]
    chosen = random.choices(qualities, weights=weights, k=1)[0]
    return chosen

################################################################################
# 5) Main generator
################################################################################

def generate_jazz_progression(length=8, allow_minor=True):
    """
    Generate a jazz chord progression in Harte notation with chord pools
    that have assigned probabilities.
    """
    # Decide if it's a minor key
    is_minor = (random.random() < 0.5) if allow_minor else False
    
    # Pick a random root
    key = random.choice(ROOTS)
    scale = build_scale(key, minor=is_minor)

    chord_pools = MINOR_CHORD_POOLS if is_minor else MAJOR_CHORD_POOLS

    # Helper to pick a chord by scale degree
    def chord_of_degree(deg):
        chord_pool = chord_pools[deg]
        chosen_quality = pick_chord_quality(chord_pool)
        # Construct Harte notation, e.g. "C:maj7"
        return f"{scale[deg - 1]}:{chosen_quality}"

    # 1. Start on I
    progression = [chord_of_degree(1)]

    # 2. Build middle chords
    for pos in range(1, length - 1):
        prev_chord = progression[-1]
        prev_degree = get_scale_degree(prev_chord, scale)

        # second-to-last => pick V for a strong cadence
        if pos == length - 2:
            progression.append(chord_of_degree(5))
            continue

        # functional rules
        if prev_degree == 1:
            # I -> random subdominant area
            next_deg = random.choice([2, 4, 6])
            progression.append(chord_of_degree(next_deg))

        elif prev_degree in (2, 4):
            # ii or IV -> V
            progression.append(chord_of_degree(5))

        elif prev_degree == 5:
            # V -> 70% I, 30% vi
            if random.random() < 0.7:
                progression.append(chord_of_degree(1))
            else:
                progression.append(chord_of_degree(6))

        elif prev_degree == 6:
            # vi -> ii
            progression.append(chord_of_degree(2))

        else:
            # fallback
            progression.append(chord_of_degree(2))

    # 3. End on I
    progression.append(chord_of_degree(1))

    return " ".join(progression)

def reformat_chord_sequence(metadata: dict, song_length: float) -> list:
    """
    Convert the chord sequence in metadata to a new format.

    Each chord is assumed to take one bar. The bar duration is computed
    using the BPM and meter information in metadata:
        bar_duration = (60 / bpm) * meter

    For each chord in the original sequence, the function computes the
    start time and duration. If a chord's full duration would exceed the
    end of the song, the duration is clipped to ensure it ends exactly at
    song_length. Chords with a start time beyond the song_length are ignored.

    Args:
        metadata (dict): A dictionary with keys 'bpm', 'meter', and 'chord_sequence'.
        song_length (float): Total length of the song in seconds.

    Returns:
        list: A sorted list of dictionaries, each containing:
            - 'value': str, the chord.
            - 'time': float, the start time in seconds.
            - 'duration': float, the duration in seconds.
    """
    bpm = metadata["bpm"]
    meter = metadata["meter"]
    chord_sequence = metadata["chord_sequence"]

    # Calculate the duration of one bar (each chord occupies one bar)
    bar_duration = (60 / bpm) * meter

    new_chord_seq = []
    for i, chord in enumerate(chord_sequence):
        start_time = i * bar_duration
        if start_time >= song_length:
            # If the chord would start after the song ends, stop processing
            break
        # Compute the duration for the chord
        duration = bar_duration
        if start_time + duration > song_length:
            # Clip the chord's duration at the end of the song.
            duration = song_length - start_time
        new_chord_seq.append({
            "value": str(chord),
            "time": start_time,
            "duration": duration
        })

    # Ensure the list is sorted by start time (this is typically already the case)
    new_chord_seq.sort(key=lambda x: x["time"])
    return new_chord_seq

if __name__ == "__main__":
    for _ in range(5):
        print(generate_jazz_progression(length=40, allow_minor=True))