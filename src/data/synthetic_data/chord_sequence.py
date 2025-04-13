import random
from src.utils import id_to_chord, NUM_CHORDS

def sample_chord_sequence(seq_length: int) -> str:
    """
    Generates a chord progression by randomly sampling chords for each bar.

    Parameters:
        seq_length (int): The number of bars (chords) to generate.

    Returns:
        str: A space-separated string representing the chord progression.
             For example: "Cmaj7 F7 Dm7 G7"
    """
    chords = []
    for _ in range(seq_length):
        # Sample a random chord ID from 2 to NUM_CHORDS-1. (2 as a minimum so that there are no 'N' or 'X' chords)
        chord_id = random.randint(2, NUM_CHORDS - 1)
        # Convert the chord ID to a chord string.
        chord = id_to_chord(chord_id)
        chords.append(chord)
    # Return the chord progression as a space-separated string.
    return " ".join(chords)
