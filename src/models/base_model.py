import autorootcwd

import torch

from src.models.hmm_smoother import HMMSmoother
from src.utils import NUM_CHORDS


class BaseACR(torch.nn.Module):
    """
    Base class for Automatic Chord Recognition models.
    """

    def __init__(self, hmm_smoothing: bool = True, hmm_alpha: float = 0.2):
        """
        Args:
            hmm_smoothing (bool): If True, the model will apply HMM smoothing to the chord predictions.
        """
        super().__init__()
        if hmm_smoothing:
            self.hmm_smoother = HMMSmoother(num_classes=NUM_CHORDS, alpha=hmm_alpha)

    def predict(self, features: torch.Tensor = None, gens: torch.Tensor = None, device = None) -> torch.Tensor:
        """
        Given a tensor of features, predict the chord labels as a one-hot tensor.

        Args:
            features (torch.Tensor): A tensor of features of shape (B, frames, features).
        Returns:
            chord_labels (torch.Tensor): A tensor of chord ids of shape (B, frames).
        """
        with torch.no_grad():
            if hasattr(self, "use_generative_features") and self.use_generative_features:
                if gens is None:
                    raise ValueError("Generative features must be provided.")
                output = self(features, gens)
            else:
                output = self(features) 
            if hasattr(self, "hmm_smoother"):
                output = self.hmm_smoother(output, device)
        return torch.argmax(output, dim=-1)

    def __str__(self):
        raise NotImplementedError()

    def to_dict(self):
        raise NotImplementedError()
