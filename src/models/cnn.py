import autorootcwd
import torch
import torch.nn as nn

from src.models.base_model import BaseACR
from src.utils import NUM_CHORDS


class CNN(BaseACR):
    """
    A simple CNN model for chord recognition, using a stack of 2D convolutions on CQT,
    and just a linear projection for generative features.
    """

    def __init__(
        self,
        input_features: int = 216,
        num_classes: int = NUM_CHORDS,
        num_layers: int = 1,
        kernel_size: int = 5,
        channels: int = 5,
        activation: str = "relu",
        hmm_smoothing: bool = True,
        hmm_alpha: float = 0.2,
        use_cqt: bool = True,
        use_generative_features: bool = False,
        gen_down_dimension: int = 256,
        gen_dimension: int = 2048,
    ):
        super().__init__(hmm_smoothing=hmm_smoothing, hmm_alpha=hmm_alpha)

        if not (use_cqt or use_generative_features):
            raise ValueError("Must use at least one of cqt or generative features.")

        self.input_features = input_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.channels = channels
        self.activation = activation
        self.use_cqt = use_cqt
        self.use_generative_features = use_generative_features
        self.gen_dimension = gen_dimension
        self.gen_down_dimension = gen_down_dimension

        act_layer = nn.ReLU if activation == "relu" else nn.PReLU

        if self.use_cqt:
            layers = []
            in_channels = 1
            for _ in range(num_layers):
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=channels,
                        kernel_size=(kernel_size, kernel_size),
                        padding="same",
                    )
                )
                layers.append(act_layer())
                in_channels = channels
            self.temporal_cnn = nn.Sequential(*layers)

            # Collapse frequency dimension with 1xF convolution
            self.freq_collapse = nn.Conv2d(
                in_channels=channels,
                out_channels=36,
                kernel_size=(1, input_features),
                padding=(0, 0)
            )

            # Final dense layer per frame
            self.classifier = nn.Linear(36, num_classes)

        if self.use_generative_features:
            self.gen_projector = nn.Linear(self.gen_dimension, self.gen_down_dimension)

        # Final classifier
        total_features = 0
        if self.use_cqt:
            total_features += 36
        if self.use_generative_features:
            total_features += self.gen_down_dimension

        self.classifier = nn.Linear(total_features, num_classes)

    def forward(
        self,
        cqt_features: torch.Tensor = None,
        gen_features: torch.Tensor = None,
    ) -> torch.Tensor:
        feature_list = []

        if self.use_cqt:
            if cqt_features is None:
                raise ValueError("CQT is enabled but no cqt_features were provided.")

            x = cqt_features.unsqueeze(1)  # (B, 1, T, F)
            x = self.temporal_cnn(x)       # (B, C, T, F)
            x = self.freq_collapse(x)      # (B, 36, T, 1)
            x = x.squeeze(-1).permute(0, 2, 1)  # (B, T, 36)
            feature_list.append(x)

        if self.use_generative_features:
            if gen_features is None:
                raise ValueError("Generative features are enabled but no gen_features were provided.")

            x_gen = self.gen_projector(gen_features)  # (B, T, gen_down_dimension)
            feature_list.append(x_gen)

        if len(feature_list) == 2:
            x = torch.cat(feature_list, dim=2)
        else:
            x = feature_list[0] 
        

        x = self.classifier(x)  # (B, T, num_classes)
        return x

    def __str__(self):
        return f"CNN(cqt:{self.use_cqt}, gen:{self.use_generative_features})"

    def to_dict(self):
        return {
            "model": "CNN",
            "input_features": self.input_features,
            "num_classes": self.num_classes,
            "num_layers": self.num_layers,
            "kernel_size": self.kernel_size,
            "channels": self.channels,
            "activation": self.activation,
            "use_cqt": self.use_cqt,
            "use_generative_features": self.use_generative_features,
            "gen_dimension": self.gen_dimension,
        }


def main():
    model = CNN(input_features=216, num_classes=25, num_layers=1, kernel_size=5)
    features = torch.randn(32, 2000, 216)  # (B, T, F)
    output = model(cqt_features=features)
    print(output.shape)  # Should be (32, 2000, 25)


if __name__ == "__main__":
    main()