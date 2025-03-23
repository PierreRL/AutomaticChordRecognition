"""
A simple logistic regression model for chord recognition.
"""

import autorootcwd

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseACR
from src.utils import NUM_CHORDS


class LogisticACR(BaseACR):
    """
    Pytorch implementation of a simple logistic regression model for Automatic Chord Recognition.
    """

    def __init__(self, input_features: int = 216, num_classes: int = NUM_CHORDS, hmm_smoothing: bool = True, hmm_alpha: float = 0.2):
        """
        Initializes the LogisticACR model.

        Args:
            input_features (int): Number of input frequency bins (e.g., 216 for CQT features).
            num_classes (int): Number of chord classes in the vocabulary.
        """
        super(LogisticACR, self).__init__(hmm_smoothing, hmm_alpha)
        self.num_classes = num_classes
        self.linear = nn.Linear(input_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        return self.linear(x)

    def __str__(self):
        return f"LogisticACR(input_features={self.linear.in_features}, num_classes={self.linear.out_features})"

    def to_dict(self):
        return {
            "model": "LogisticACR",
            "input_features": self.linear.in_features,
            "num_classes": self.linear.out_features,
        }
