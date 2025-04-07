"""
Custom structured loss for Chord Recognition.

Loss is over the root and set of pitch classes, as well as the overall chord.
The root and pitch class loss are combined into a single value.
This is then combined in a convex combination with the overall chord loss.

Weights over classes can also be applied.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from src.utils import get_chord_root_batch, get_pitch_classes_batch, get_torch_device


class StructuredLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, ignore_index: int = -1, class_weights: Optional[Tensor] = None, ignore_X_pitches: bool = True):
        """
        Args:
            alpha (float): Weight for the root and pitch class loss.
            ignore_index (int): Index to ignore in the target tensor.
            class_weights (Tensor, optional): Weights for each class. If None, no weights are applied.
            ignore_X_pitches (bool): Whether to ignore "X" pitches in the pitch class and root loss.
        """
        super().__init__()
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.class_weights = class_weights
        self.ignore_X_pitches = ignore_X_pitches

    def forward(self, output: Tuple[Tensor, Tensor, Tensor], target: Tensor, device = None) -> Tensor:
        """
        Compute the structured loss.

        Args:
            output: Tuple of (chord_output, root_output, pitch_class_output),
                    each of shape (B*T, num_classes).
            target: Tensor of shape (B*T,), with chord class IDs.

        Returns:
            Tensor: Scalar loss value.
        """
        if device is None:
            device = get_torch_device()

        chord_output, root_output, pitch_class_output = output

        # Overall chord loss
        chord_loss = F.cross_entropy(
            chord_output, target, ignore_index=self.ignore_index
        )

        # Mask out invalid targets (and optional "X" pitches)
        valid_mask = target != self.ignore_index  # (B*T,)
        if self.ignore_X_pitches:
            valid_mask &= (target != 1)  # chord ID 1 is "X" (no chord), skip it too
        valid_targets = target[valid_mask]  # Only valid chord class IDs

        # Root loss
        root_output_valid = root_output[valid_mask]  # (N_valid, num_root_classes)
        root = get_chord_root_batch(valid_targets).long().to(device)  # (N_valid,)
        root_loss = F.cross_entropy(
            root_output_valid,
            root,
        )

        # Pitch class loss
        valid_pitch_class_output = pitch_class_output[valid_mask]  # Shape: (N_valid, 12)
        pitch_classes = get_pitch_classes_batch(valid_targets).float().to(device)  # Shape: (N_valid, 12)
        pitch_class_loss = F.binary_cross_entropy_with_logits(
            valid_pitch_class_output, pitch_classes, reduction='mean'
        )

        # Weighted convex combination
        combined_loss = (
            (1 - self.alpha) * chord_loss +
            self.alpha * (root_loss + pitch_class_loss)
        )

        return combined_loss

    def __repr__(self):
        return f"StructuredLoss(alpha={self.alpha}, ignore_index={self.ignore_index})"
