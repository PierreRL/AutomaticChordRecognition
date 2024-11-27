import autorootcwd
from torch.utils.data import random_split
import torch

from src.models.ismir2017 import ISMIR2017ACR
from src.data.dataset import FullChordDataset
from src.evaluation.evaluation import evaluate_model
from src.utils import get_torch_device


def main():
    dataset = FullChordDataset()

    model = ISMIR2017ACR(input_features=dataset.n_bins, num_classes=25)
    model.load_state_dict(torch.load("./data/models/best_model.pth", weights_only=True))

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    metrics = evaluate_model(model, val_dataset, batch_size=32)

    print(metrics)


if __name__ == "__main__":
    main()
