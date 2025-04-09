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

    def predict(
        self,
        features: torch.Tensor = None,
        gens: torch.Tensor = None,
        mask: torch.Tensor = None,
        device=None
    ) -> torch.Tensor:
        """
        Predict chord class indices from input features.

        Args:
            features (torch.Tensor): Input features of shape (B, frames, features).
            gens (torch.Tensor): Optional generative features of shape (B, frames, gen_dim).
            mask (torch.Tensor): Boolean mask of valid time steps for CRF decoding, shape (B, frames).
            device (torch.device): The device to run prediction on.

        Returns:
            torch.Tensor: Predictions of shape (B, frames), padded with -1 if CRF is used.
        """
        with torch.no_grad():
            if hasattr(self, "use_generative_features") and self.use_generative_features:
                if gens is None:
                    raise ValueError("Generative features must be provided for this model.")
                output = self(features, gens)
            else:
                output = self(features)

            if hasattr(self, "structured_loss") and self.structured_loss:
                output = output[0]

            if hasattr(self, "crf"):
                if mask is None:
                    raise ValueError("CRF decoding requires a mask to be provided.")
                predictions = self.crf.decode(output, mask=mask)  # List[List[int]]

                # Pad to rectangular tensor (with -1s)
                max_len = output.size(1)
                pred_tensor = torch.full((len(predictions), max_len), fill_value=-1, device=output.device)
                for i, seq in enumerate(predictions):
                    pred_tensor[i, :len(seq)] = torch.tensor(seq, device=output.device)
                return pred_tensor

            if hasattr(self, "hmm_smoother"):
                output = self.hmm_smoother(output, device)

            return torch.argmax(output, dim=-1)

    def __str__(self):
        raise NotImplementedError()

    def to_dict(self):
        raise NotImplementedError()
