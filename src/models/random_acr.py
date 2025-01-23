import autorootcwd
import torch

from src.models.base_model import BaseACR
from src.utils import NUM_CHORDS


class RandomACR(BaseACR):
    """
    A baseline that randomly picks a chord classes.
    """

    def __init__(self, chord_vocab_size: int = NUM_CHORDS, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chord_vocab_size = chord_vocab_size

    def forward(self, features):
        num_batches, num_frames, _ = features.shape
        labels = torch.randint(0, self.chord_vocab_size, (num_batches, num_frames))
        return torch.nn.functional.one_hot(labels, self.chord_vocab_size).float()
