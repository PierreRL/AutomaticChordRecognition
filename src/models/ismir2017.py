"""
A pytorch implementation of the CNNRNN model proposed in the ISMIR 2017 paper:
STRUCTURED TRAINING FOR LARGE-VOCABULARY CHORD
RECOGNITION

https://brianmcfee.net/papers/ismir2017_chord.pdf
"""

import autorootcwd

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseACR
from src.utils import NUM_CHORDS


class ISMIR2017ACR(BaseACR):
    """
    Pytorch implementation of the ISMIR 2017 model for Automatic Chord Recognition.
    """

    def __init__(
        self, input_features: int = 216, num_classes: int = NUM_CHORDS, cr2: bool = True
    ):
        """
        Initializes the ISMIR2017ACR model.

        Args:
            input_features (int): Number of input frequency bins (e.g., 216 for CQT features).
            num_classes (int): Number of chord classes in the vocabulary.
            cr2 (bool): If True, the model uses the CR2 variant, cr1 otherwise.
        """
        super(ISMIR2017ACR, self).__init__()
        self.cr2 = cr2

        self.batch_norm = nn.BatchNorm2d(
            1
        )  # Normalize input along the channel dimension

        # 5x5 convolution layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(5, 5),
            padding="same",  # Keep output size same as input
        )

        # 1xF convolution (full-height filter bank with 36 filters)
        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=36,
            kernel_size=(1, input_features),  # Full-height filter
            padding=(0, 0),  # Valid padding to reduce frequency dimension to 1
        )

        encoder_hidden_size = 128 if self.cr2 else 256

        # Encoder: Bidirectional GRU layer
        self.bi_gru_encoder = nn.GRU(
            input_size=36,
            hidden_size=encoder_hidden_size,  # E per direction, output 2*E
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        if cr2:
            # Decoder: Secondary Bidirectional GRU for CR2
            self.bi_gru_decoder = nn.GRU(
                input_size=2 * encoder_hidden_size,  # 2*E from encoder
                hidden_size=128,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

        # Dense output layer
        self.dense = nn.Linear(
            256 if self.cr2 else 512, num_classes
        )  # GRU outputs 256 if cr2, 512 if cr1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISMIR2017ACR model.

        Args:
            features (torch.Tensor): Input tensor of shape (B, frames, features).

        Returns:
            torch.Tensor: Output tensor of shape (B, frames, num_classes).
        """
        # Input is (B, frames, features). Add a channel dimension to make it (B, 1, frames, features)
        x = features.unsqueeze(1)

        # Batch normalization
        x = self.batch_norm(x)  # (B, 1, frames, features)

        # First convolutional layer (5x5)
        x = F.relu(self.conv1(x))  # (B, 1, frames, features)

        # Second convolutional layer (1xF) with 36 filters
        x = F.relu(self.conv2(x))  # (B, 36, frames, 1)

        # Remove frequency dimension
        x = x.squeeze(-1)  # (B, 36, frames)

        # Permute to (B, frames, 36) for GRU
        x = x.permute(0, 2, 1)  # (B, frames, 36)

        # Bidirectional GRU layer
        x, _ = self.bi_gru_encoder(x)  # (B, frames, 2*E)

        if self.cr2:
            # Secondary Bidirectional GRU layer
            x, _ = self.bi_gru_decoder(x)  # (B, frames, 256)

        # Dense output layer with softmax activation over chord classes
        x = self.dense(x)  # (B, frames, num_classes)

        return x

    def __str__(self):
        return f"ISMIR2017ACR(cr2:{self.cr2})"


def main():
    model = ISMIR2017ACR(input_features=216, num_classes=25)
    features = torch.randn(32, 2000, 216)  # (B, frames, features)
    output = model(features)
    print(output.shape)


if __name__ == "__main__":
    main()
