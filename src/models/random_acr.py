import autorootcwd
from src.models.base_model import BaseACR
import torch


class RandomACR(BaseACR):
    """
    A baseline that randomly picks a chord classes.
    """

    def __init__(self, chord_vocab_size: int = 25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chord_vocab_size = chord_vocab_size

    def forward(self, features):
        num_batches, num_frames, _ = features.shape
        vocab_size = 25
        labels = torch.randint(0, self.chord_vocab_size, (num_batches, num_frames))
        return torch.nn.functional.one_hot(labels, vocab_size).float()
