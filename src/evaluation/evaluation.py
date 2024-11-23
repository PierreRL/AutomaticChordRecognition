import autorootcwd
from tqdm import tqdm
from enum import Enum

import torch
from torch.utils.data import DataLoader
import mir_eval

from src.models.base_model import BaseACR
from src.data.dataset import FullChordDataset
from src.utils import id_to_chord, get_torch_device


class EvalMetric(Enum):
    """
    Defines an ENUM of evaluation metrics used in this project from the mir_eval library.

    Attributes:
        ROOT: Chord root evaluation metrics.
        MAJMIN: Major/minor evaluation metrics.
        MIREX: MIREX evaluation metric - returns 1 if at least three pitches are in common.
        THIRD: Third evaluation metric - returns 1 if the third is in common.
        SEVENTH: Seventh evaluation metric - returns 1 if the seventh is in common.
    """

    def __str__(self):
        return self.name

    ROOT = "root"
    MAJMIN = "majmin"
    MIREX = "mirex"
    THIRD = "third"
    SEVENTH = "seventh"

    def eval_func(self) -> callable:
        """
        Returns the evaluation function for this evaluation metric.
        """
        if self == EvalMetric.ROOT:
            return mir_eval.chord.root
        elif self == EvalMetric.MAJMIN:
            return mir_eval.chord.majmin
        elif self == EvalMetric.MIREX:
            return mir_eval.chord.mirex
        elif self == EvalMetric.THIRD:
            return mir_eval.chord.thirds
        elif self == EvalMetric.SEVENTH:
            return mir_eval.chord.sevenths
        else:
            raise ValueError(f"Invalid evaluation metric: {self}")

    def evaluate(self, hypotheses: torch.Tensor, references: torch.Tensor):
        """
        Evaluate a model on a dataset split using a list of evaluation metrics.

        Args:
            hypotheses (torch.Tensor): The model's chord predictions as ids. Shape (B, frames)
            references (torch.Tensor): The ground truth chord labels as ids. Shape (B, frames)

        Returns:
            metrics (torch.Tensor): A tensor of evaluation metrics and their values. Shape (B, frames)
        """

        # Initialize the evaluation metrics tensor. Shape (num_batches, num_frames)
        metrics = torch.zeros_like(hypotheses, dtype=torch.float32)

        # Iterate over the batch of chord predictions and ground truth labels
        ref_labels = []
        hyp_labels = []
        for i in range(hypotheses.shape[0]):
            # Convert the chord labels from indices to strings
            ref_labels.extend([id_to_chord(id) for id in references[i]])
            hyp_labels.extend([id_to_chord(id) for id in hypotheses[i]])

        # Evaluate the chord labels using the evaluation metric
        metrics = torch.from_numpy(self.eval_func()(ref_labels, hyp_labels))

        # Reshape the metrics tensor to include the batch dimension again
        metrics = metrics.reshape(hypotheses.shape)

        return metrics


def evaluate_model(
    model: BaseACR,
    dataset: FullChordDataset,
    evals: list[EvalMetric] = [
        EvalMetric.ROOT,
        EvalMetric.MAJMIN,
        EvalMetric.MIREX,
        EvalMetric.THIRD,
        EvalMetric.SEVENTH,
    ],
    batch_size: int = 64,
    device: torch.device = None,
) -> dict[str, float]:
    """
    Evaluate a model on a dataset split using a list of evaluation metrics.

    Args:
        model (BaseACRModel): The model to evaluate.
        dataset (ChordDataset): The dataset to evaluate on.
        evals (list[EvalMetrics]): The evaluation metrics to use. Defaults to [EvalMetrics.ROOT, EvalMetrics.MAJMIN, EvalMetrics.MIREX, EvalMetrics.CHORD_OVERLAP, EvalMetrics.CHORD_LABEL].
        batch_size (int): The batch size to use for evaluation. Defaults to 64.
        device (torch.device): The device to use for evaluation. Defaults to None.

    Returns:
        metrics (dict[str, float]): A dictionary of evaluation metrics and their values.
    """

    if not device:
        device = get_torch_device()

    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Initialize the evaluation metrics dictionary
    metrics = {}

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for eval in evals:
        # Initialize the evaluation metric
        metrics[eval.value] = 0.0

    print("Evaluating model...")
    # Evaluate the model on the data loader
    for batch_features, batch_labels in tqdm(data_loader):
        # Send the batch to the device
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(
            device
        )

        # Get the chord predictions from the model
        predictions = model.predict(batch_features)

        # Evaluate the model on the sample using the evaluation metrics
        for eval in evals:
            metrics[eval.value] += torch.mean(eval.evaluate(predictions, batch_labels))

    # Calculate the average evaluation metrics
    for eval in evals:
        metrics[eval.value] /= len(dataset)

    return metrics


def main():

    from torch.utils.data import random_split
    from src.models.random_acr import RandomACR

    dataset = FullChordDataset()

    # Split the dataset into train and test
    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size

    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Evaluate the random model on the test dataset
    model = RandomACR()

    metrics = evaluate_model(model, test_dataset)
    print(metrics)


if __name__ == "__main__":
    main()
