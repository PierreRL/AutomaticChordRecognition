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
        ("maj", 0.3),
        ("maj7", 0.4),
        ("maj6", 0.2),
        ("sus4", 0.1)
    ],
    # ii
    2: [
        ("min", 0.3),
        ("min7", 0.4),
        ("dim", 0.1),
        ("hdim7", 0.1),
        ("sus2", 0.1)
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
        ("hdim7", 0.3),
        ("sus2", 0.4)
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
        ("min7", 0.4),
        ("dim", 0.2)
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

def generate_jazz_progression(seq_length=None, allow_minor=True):
    """
    Generate a jazz chord progression in Harte notation with chord pools
    that have assigned probabilities.
    """
    if seq_length is None:
        seq_length = random.randint(4, 10)
    # Decide if it's a minor key
    is_minor = (random.random() < 0.5) if allow_minor else False
    
    # Pick a random root
    key = random.choice(ROOTS)
    scale = build_scale(key, minor=is_minor)

    chord_pools = MINOR_CHORD_POOLS if is_minor else MAJOR_CHORD_POOLS

    # Helper to pick a chord by scale degree
    # Pre-select chords for each scale degree
    degree_to_chord = {}

    for deg in range(1, 8):
        chord_pool = chord_pools[deg]
        chosen_quality = pick_chord_quality(chord_pool)
        degree_to_chord[deg] = f"{scale[deg - 1]}:{chosen_quality}"

    def chord_of_degree(deg):
        return degree_to_chord[deg]

    # 1. Start on I
    progression = [chord_of_degree(1)]

    # 2. Build middle chords
    for pos in range(1, seq_length-1):
        prev_chord = progression[-1]
        prev_degree = get_scale_degree(prev_chord, scale)

        # second-to-last => pick V for a strong cadence
        # if pos == seq_length - 2:
        #     progression.append(chord_of_degree(5))
        #     continue

        # functional rules
        if prev_degree == 1:
            next_deg = random.choices([2, 4, 6, 3], weights=[0.3, 0.3, 0.3, 0.1])[0]
            # I -> ii, IV, vi, or iii with smaller chance
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
            # vi -> ii or iii
            next_deg = random.choices([2, 3], weights=[0.8, 0.2])[0]
            progression.append(chord_of_degree(next_deg))

        elif prev_degree == 3:
            progression.append(chord_of_degree(6))  # iii → vi

        else:
            # fallback
            progression.append(chord_of_degree(2))

    # 3. End on V
    progression.append(chord_of_degree(5))

    return " ".join(progression)

def reformat_chord_sequence(metadata: dict, song_length: float) -> list:
    """
    Convert the chord sequence in metadata to a repeated time-aligned format.

    Each chord is assumed to take one bar. The bar duration is computed
    using the BPM and meter information in metadata:
        bar_duration = (60 / bpm) * meter

    The chord sequence is repeated as necessary to fill up to the song_length.
    If the final chord would overrun the song, its duration is clipped.
    Chords with a start time beyond the song_length are ignored.

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
    chord_sequence = metadata["chord_sequence"].split(" ")

    bar_duration = (60 / bpm) * meter
    new_chord_seq = []

    i = 0
    while True:
        start_time = i * bar_duration
        if start_time >= song_length:
            break

        chord = chord_sequence[i % len(chord_sequence)]
        duration = min(bar_duration, song_length - start_time)

        new_chord_seq.append({
            "value": str(chord),
            "time": start_time,
            "duration": duration
        })
        i += 1

    return new_chord_seq

if __name__ == "__main__":
    for _ in range(20):
        print(generate_jazz_progression(seq_length=8, allow_minor=True))