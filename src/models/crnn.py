"""
A pytorch implementation of the CNNRNN model proposed in the ISMIR 2017 paper:
STRUCTURED TRAINING FOR LARGE-VOCABULARY CHORD
RECOGNITION

https://brianmcfee.net/papers/crnn_chord.pdf
"""

import autorootcwd

import torch
import torch.nn as nn
from torchcrf import CRF

from src.models.base_model import BaseACR
from src.utils import NUM_CHORDS, get_torch_device


class CRNN(BaseACR):
    """
    Pytorch implementation of the ISMIR 2017 model for Automatic Chord Recognition,
    extended to optionally handle generative features in addition to or instead of CQT.
    """

    def __init__(
        self,
        input_features: int = 216,
        num_classes: int = NUM_CHORDS,
        hidden_size: int = 201,
        num_layers: int = 1,
        cnn_layers: int = 1,
        kernel_size: int = 5,
        cnn_channels: int = 1,
        cr2: bool = False,
        activation: str = "prelu",
        structured_loss: bool = False,
        use_cqt: bool = True,
        input_transitions: bool = False,
        use_generative_features: bool = False,
        gen_down_dimension: int = 128,
        gen_dimension: int = 2048,
        hmm_smoothing: bool = False,
        hmm_alpha: float = 0.2,
        crf: bool = False,
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
            hmm_smoothing (bool): Whether to apply HMM smoothing at inference.
            hmm_alpha (float): Smoothing alpha for the HMM.
            use_cqt (bool): If True, use the standard CRNN convolutional pipeline on the CQT.
            use_generative_features (bool): If True, accept external generative features that
                skip the convolution layers and go straight into the RNN.
            gen_down_dimension (int): Dimensionality of the generative feature vector after projection.
            gen_dimension (int): Dimensionality of each generative feature vector per frame.
        """
        super().__init__(hmm_smoothing=hmm_smoothing, hmm_alpha=hmm_alpha)

        # Must use at least one of the two feature types
        if not (use_cqt or use_generative_features):
            raise ValueError("Must use at least one of cqt or generative features.")

        self.cr2 = cr2
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.cnn_layers = cnn_layers
        self.kernel_size = kernel_size
        self.cnn_channels = cnn_channels
        self.activation = activation
        self.use_cqt = use_cqt
        self.input_transitions = input_transitions
        self.use_generative_features = use_generative_features
        self.gen_dimension = gen_dimension
        self.gen_down_dimension = gen_down_dimension
        self.structured_loss = structured_loss
        if crf:
            self.crf = CRF(num_classes, batch_first=True)

        if self.activation not in ["relu", "prelu"]:
            raise ValueError(f"Invalid activation function: {self.activation}")
        act_layer = nn.ReLU if activation == "relu" else nn.PReLU
        #
        # ----- Convolutional pipeline (used only if use_cqt=True) -----
        #
        if self.use_cqt:
            self.batch_norm = nn.BatchNorm2d(1)
            layers = []
            in_channels = 1
            for _ in range(cnn_layers):
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=cnn_channels,
                        kernel_size=(kernel_size, kernel_size),
                        padding="same",
                    )
                )
                layers.append(act_layer())
                in_channels = cnn_channels
            self.temporal_cnn = nn.Sequential(*layers)

            self.freq_collapse = nn.Conv2d(
                in_channels=cnn_channels,
                out_channels=36,
                kernel_size=(1, self.input_features),
                padding=(0, 0),
            )

        if self.use_generative_features:
            self.gen_projector = nn.Linear(self.gen_dimension, self.gen_down_dimension)
            self.gen_norm = nn.LayerNorm(self.gen_down_dimension)

        # ----- RNN input dimension -----
        rnn_input_dim = 0
        if self.use_cqt:
            rnn_input_dim += 36
        if self.use_generative_features:
            rnn_input_dim += self.gen_down_dimension
        if self.input_transitions:
            rnn_input_dim += 1

        encoder_hidden_size = self.hidden_size // 2 if self.cr2 else self.hidden_size

        # Bidirectional GRU "encoder"
        self.bi_gru_encoder = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=encoder_hidden_size,  # E per direction, total 2E out
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        if self.cr2:
            # Optional second GRU "decoder"
            self.bi_gru_decoder = nn.GRU(
                input_size=2 * encoder_hidden_size,  # 2E from encoder
                hidden_size=encoder_hidden_size,  # E per direction
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=True,
            )

        # Final dense layer
        out_dim = encoder_hidden_size if cr2 else 2 * encoder_hidden_size

        if self.structured_loss:
            self.root_dense = nn.Linear(out_dim, 14)  # 12 for root, 2 for "X" and "N"
            self.pitch_class_dense = nn.Linear(out_dim, 12)
            out_dim += (
                14 + 12
            )  # We concat the root and pitch class outputs to the final output

        self.dense = nn.Linear(out_dim, num_classes)

    def forward(
        self,
        cqt_features: torch.Tensor = None,
        gen_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the CRNN model.

        Args:
            cqt_features (torch.Tensor): If use_cqt is True, shape (B, frames, freq_bins).
            gen_features (torch.Tensor): If use_generative_features is True, shape (B, frames, gen_dimension).

        Returns:
            torch.Tensor: Output of shape (B, frames, num_classes).
        """

        if self.input_transitions:
            # The last channel of cqt_features is the transitions
            transitions = cqt_features[:, :, -1]  # (B, frames)
            cqt_features = cqt_features[:, :, :-1]

        # Collect the features to feed into the RNN
        feature_list = []

        # CQT pipeline
        if self.use_cqt:
            if cqt_features is None:
                raise ValueError("CQT is enabled but no cqt_features were provided.")

            # (B, frames, freq) -> (B, 1, frames, freq)
            x_cqt = cqt_features.unsqueeze(1)
            x_cqt = self.batch_norm(x_cqt)
            x_cqt = self.temporal_cnn(x_cqt)
            x_cqt = self.freq_collapse(x_cqt)
            x_cqt = x_cqt.squeeze(-1).permute(0, 2, 1)  # (B, frames, 36)
            feature_list.append(x_cqt)

        # Generative features
        if self.use_generative_features:
            if gen_features is None:
                raise ValueError(
                    "Generative features are enabled but no gen_features were provided."
                )
            # Suppose gen_features is shape (B, frames, gen_dimension)
            # We skip convolution entirely. Just add it.
            gen_features = self.gen_projector(
                gen_features
            )  # (B, frames, gen_down_dimension)
            gen_features = self.gen_norm(
                gen_features
            )  # (B, frames, gen_down_dimension)
            feature_list.append(gen_features)

        # Combine features if we have both
        if len(feature_list) == 2:
            x = torch.cat(feature_list, dim=2)  # (B, frames, 36 + gen_down_dimension)
        else:
            x = feature_list[0]  # Either cqt alone or generative alone

        if self.input_transitions:
            # Add the transitions to the input
            x = torch.cat((x, transitions.unsqueeze(2)), dim=2)
            # (B, frames, 36 + gen_down_dimension + 1)

        x, _ = self.bi_gru_encoder(x)  # (B, frames, 2E)

        if self.cr2:
            x, _ = self.bi_gru_decoder(x)  # (B, frames, 2E) again

        if self.structured_loss:
            # Compute root and pitch class outputs
            root_out = self.root_dense(x)
            pitch_class_out = self.pitch_class_dense(x)
            # Concatenate root and pitch class outputs to the final output
            x = torch.cat(
                (x, root_out, pitch_class_out), dim=2
            )  # (B, frames, 2E + NUM_CHORDS + 12)

        x = self.dense(x)  # (B, frames, num_classes)

        if self.structured_loss:
            # Return the root and pitch class outputs separately
            return x, root_out, pitch_class_out

        return x

    def __str__(self):
        return f"CRNN(cr2:{self.cr2}, cqt:{self.use_cqt}, gen:{self.use_generative_features})"

    def to_dict(self):
        return {
            "model": "CRNN",
            "input_features": self.input_features,
            "num_classes": self.num_classes,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "cnn_layers": self.cnn_layers,
            "kernel_size": self.kernel_size,
            "cnn_channels": self.cnn_channels,
            "crf": hasattr(self, "crf"),
            "hmm_smoothing": hasattr(self, "hmm_smoother"),
            "hmm_alpha": (
                self.hmm_smoother.hmm_alpha if hasattr(self, "hmm_smoother") else None
            ),
            "structured_loss": self.structured_loss,
            "cr2": self.cr2,
            "activation": self.activation,
            "use_cqt": self.use_cqt,
            "use_generative_features": self.use_generative_features,
            "gen_dimension": self.gen_dimension,
            "gen_down_dimension": self.gen_down_dimension,
            "structured_loss": self.structured_loss,
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
