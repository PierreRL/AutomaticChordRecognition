import torch


class BaseACR(torch.nn.Module):
    """
    Base class for Automatic Chord Recognition models.
    """

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Given a tensor of features, predict the chord labels as a one-hot tensor.

        Args:
            features (torch.Tensor): A tensor of features of shape (B, frames, features).
        Returns:
            chord_labels (torch.Tensor): A tensor of chord ids of shape (B, frames).
        """
        output = self(features)
        return torch.argmax(output, dim=-1)

    def __str__(self):
        raise NotImplementedError()
