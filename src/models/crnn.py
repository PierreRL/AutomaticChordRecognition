"""
A pytorch implementation of the CNNRNN model proposed in the ISMIR 2017 paper:
STRUCTURED TRAINING FOR LARGE-VOCABULARY CHORD
RECOGNITION

https://brianmcfee.net/papers/crnn_chord.pdf
"""

import autorootcwd

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseACR
from src.utils import NUM_CHORDS, get_torch_device


class CRNN(BaseACR):
    """
    Pytorch implementation of the ISMIR 2017 model for Automatic Chord Recognition.
    https://brianmcfee.net/papers/crnn_chord.pdf
    """

    def __init__(
        self,
        input_features: int = 216,
        num_classes: int = NUM_CHORDS,
        hidden_size: int = 201,
        num_layers: int = 1,
        cr2: bool = False,
        crf: bool = False,
        activation: str = "relu",
        hmm_smoothing: bool = True,
        hmm_alpha: float = 0.2,
    ):
        """
        Initializes the CRNN model.

        Args:
            input_features (int): Number of input frequency bins (e.g., 216 for CQT features).
            num_classes (int): Number of chord classes in the vocabulary.
            cr2 (bool): If True, the model uses the CR2 variant, cr1 otherwise.
            hidden_size (int): Hidden size of the GRU layers.
            num_layers (int): Number of layers in the GRU.
            activation (str): Activation function to use (relu or prelu).
        """
        super().__init__(hmm_smoothing=hmm_smoothing, hmm_alpha=hmm_alpha)
        self.cr2 = cr2
        self.crf = crf
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.activation = activation
        if self.activation not in ["relu", "prelu"]:
            raise ValueError(f"Invalid activation function: {self.activation}")

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
        self.activation1 = nn.ReLU() if activation == "relu" else nn.PReLU()

        # 1xF convolution (full-height filter bank with 36 filters)
        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=36,
            kernel_size=(1, self.input_features),  # Full-height filter
            padding=(0, 0),  # Valid padding to reduce frequency dimension to 1
        )
        self.activation2 = nn.ReLU() if activation == "relu" else nn.PReLU()

        encoder_hidden_size = self.hidden_size // 2 if self.cr2 else self.hidden_size

        # Encoder: Bidirectional GRU layer
        self.bi_gru_encoder = nn.GRU(
            input_size=36,
            hidden_size=encoder_hidden_size,  # E per direction, output 2*E
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )

        if cr2:
            # Decoder: Secondary Bidirectional GRU for CR2
            self.bi_gru_decoder = nn.GRU(
                input_size=2 * encoder_hidden_size,  # 2*E from encoder
                hidden_size=encoder_hidden_size,  # E per direction
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=True,
            )

        # Dense output layer
        self.dense = nn.Linear(
            encoder_hidden_size if self.cr2 else 2 * encoder_hidden_size, num_classes
        )  # GRU outputs 256 if cr2, 512 if cr1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CRNN model.

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
        x = self.activation1(self.conv1(x))  # (B, 1, frames, features)

        # Second convolutional layer (1xF) with 36 filters
        x = self.activation2(self.conv2(x))  # (B, 36, frames, 1)

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
        return f"CRNN(cr2:{self.cr2})"

    def to_dict(self):
        return {
            "model": "CRNN",
            "input_features": self.input_features,
            "num_classes": self.num_classes,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "cr2": self.cr2,
            "activation": self.activation,
        }


def get_model(
    exp_dir="weight_alpha_search/weight_alpha_0.55",
    # exp_dir="hparams_random/segment_length_28_layers_1_hidden_size_201",
    hmm=True,
    hmm_alpha=0.2,
    device=None,
):
    if device is None:
        device = get_torch_device()
    state_dict = torch.load(
        f"./results/{exp_dir}/best_model.pth", map_location=device, weights_only=True
    )
    model = CRNN(hmm_smoothing=hmm, hmm_alpha=hmm_alpha)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def main():
    model = CRNN(input_features=216, num_classes=25)
    features = torch.randn(32, 2000, 216)  # (B, frames, features)
    output = model(features)
    print(output.shape)


if __name__ == "__main__":
    main()
