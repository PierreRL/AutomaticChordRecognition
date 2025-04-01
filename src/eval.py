import autorootcwd
from tqdm import tqdm
from enum import Enum
import os
from typing import List, Dict

import torch
import numpy as np
from torch.utils.data import DataLoader
import mir_eval
from sklearn.metrics import accuracy_score

from src.models.crnn import CRNN
from src.models.base_model import BaseACR
from src.models.hmm_smoother import HMMSmoother
from src.data.dataset import FullChordDataset
from src.utils import (
    id_to_chord_table,
    get_torch_device,
    collate_fn,
    NUM_CHORDS,
    SMALL_VOCABULARY,
    chord_to_id_map,
)


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
    SONG_WISE_ACC = "song_wise_acc"

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
        elif self == EvalMetric.SONG_WISE_ACC:
            return accuracy_score
        else:
            raise ValueError(f"Invalid evaluation metric: {self}")

    def get_eval_input_type(self) -> str:
        """
        Returns the evaluation type for this evaluation metric.
        """
        # Only SONG_WISE_ACC uses integer inputs
        if self in [EvalMetric.SONG_WISE_ACC]:
            return "int"
        else:
            return "str"

    def evaluate(self, hypotheses: torch.Tensor, references: torch.Tensor):
        """
        Evaluate a model on a dataset split using a list of evaluation metrics.

        Args:
            hypotheses (list[str]): List of strings of chord predictions.
            references (list[str]): List of strings of ground truth chord labels.

        Returns:
            metrics (nd.array): A numpy array of evaluation metrics and their values. Flatten across batches. Shape (num_frames,)
        """

        # Evaluate the chord labels using the evaluation metric
        metrics = self.eval_func()(references, hypotheses)

        # Ensure the output is always an array
        if np.isscalar(metrics):
            metrics = np.array([metrics])  # Convert scalar to 1D array

        return metrics


def evaluate_model(
    model: BaseACR,
    dataset: FullChordDataset,
    evals: List[EvalMetric] = [
        EvalMetric.ROOT,
        EvalMetric.MAJMIN,
        EvalMetric.MIREX,
        EvalMetric.THIRD,
        EvalMetric.SEVENTH,
        EvalMetric.SONG_WISE_ACC,
    ],
    batch_size: int = 64,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset split using a list of evaluation metrics.

    Args:
        model (BaseACRModel): The model to evaluate.
        dataset (FullChordDataset): The dataset to evaluate on.
        evals (list[EvalMetrics]): The evaluation metrics to use. Defaults to [EvalMetrics.ROOT, EvalMetrics.MAJMIN, EvalMetrics.MIREX, EvalMetrics.THIRD, EvalMetrics.SEVENTH].
        batch_size (int): The batch size to use for evaluation. Defaults to 32.
        device (torch.device): The device to use for evaluation. Defaults to None.

    Returns:
        metrics (dict[str, float]): A dictionary of evaluation metrics and their values.
    """
    torch.set_grad_enabled(False)  # Disable gradient calculation for evaluation
    if not device:
        device = get_torch_device()

    model.to(device)
    model.eval()

    # Initialize metrics storage
    metrics = {"mean": {}, "median": {}}

    for eval in evals:
        metrics["mean"][eval.value] = 0.0
        metrics["median"][eval.value] = []

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    song_metrics = {eval.value: {} for eval in evals}

    all_hypotheses = []
    all_references = []

    for batch_cqts, batch_gens, batch_labels in tqdm(data_loader):
        batch_cqts, batch_gens, batch_labels = batch_cqts.to(device), batch_gens.to(device), batch_labels.to(
            device
        )

        if hasattr(model, "use_generative_features") and model.use_generative_features:
            predictions = model.predict(
                batch_cqts, batch_gens, device=device
            ).to(device)
        else:
            predictions = model.predict(batch_cqts).to(device)

        for i in range(batch_labels.shape[0]):  # Iterate over songs in the batch
            if SMALL_VOCABULARY:
                valid_mask = batch_labels[i] != -1
            else:
                valid_mask = torch.logical_and(
                    batch_labels[i] != -1, batch_labels[i] != chord_to_id_map["X"]
                )
            filtered_references = batch_labels[i][valid_mask].cpu().numpy()
            filtered_hypotheses = predictions[i][valid_mask].cpu().numpy()

            all_hypotheses.extend(filtered_hypotheses)
            all_references.extend(filtered_references)

            ref_labels = id_to_chord_table[filtered_references]
            hyp_labels = id_to_chord_table[filtered_hypotheses]

            for eval in evals:
                if eval.get_eval_input_type() == "int":
                    song_eval_scores = eval.evaluate(
                        filtered_references, filtered_hypotheses
                    )
                else:
                    song_eval_scores = eval.evaluate(hyp_labels, ref_labels)

                # Filter out invalid scores that are -1 (produced by mir_eval with 'X' labels for example)
                song_eval_scores = song_eval_scores[song_eval_scores != -1]

                if i not in song_metrics[eval.value]:
                    song_metrics[eval.value][i] = []

                song_metrics[eval.value][i].append(np.mean(song_eval_scores))

    for eval in evals:
        song_scores = [np.mean(scores) for scores in song_metrics[eval.value].values()]

        metrics["mean"][eval.value] = np.mean(song_scores)
        metrics["median"][eval.value] = np.median(song_scores)

    # Flatten along song dimension
    all_hypotheses = np.array(all_hypotheses)
    all_references = np.array(all_references)

    # Compute frame-wise accuracy
    metrics["frame_wise_acc"] = accuracy_score(all_references, all_hypotheses)

    # Compute class-wise accuracy
    class_accs = np.full(NUM_CHORDS, np.nan)
    for i in range(NUM_CHORDS):
        # Find all references of class i
        mask = all_references == i
        class_references = all_references[mask]
        class_hypotheses = all_hypotheses[mask]

        if class_references.size > 0:
            class_accs[i] = accuracy_score(class_references, class_hypotheses)

        metrics["class_wise_acc_mean"] = np.nanmean(class_accs)  # Ignore NaNs
        metrics["class_wise_acc_median"] = np.nanmedian(class_accs)

    return metrics


def main():
    # Testing the seventh chords on C major
    # chord1 = "C:maj7"
    # chord2 = "C:maj"
    # chord3 = "C:min"
    # chord4 = "C:min7"
    # chord5 = "C:7"
    # chord6 = "C:dim"

    # # print(EvalMetric.ROOT.evaluate([chord1], [chord2]))
    # # print(EvalMetric.ROOT.evaluate([chord1], [chord3]))
    # print(EvalMetric.SEVENTH.evaluate([chord1], [chord1]))
    # print(EvalMetric.SEVENTH.evaluate([chord1], [chord2]))
    # print(EvalMetric.SEVENTH.evaluate([chord2], [chord1]))

    from torch.utils.data import random_split

    full_length_dataset = FullChordDataset()

    # Split the dataset into train and test
    train_size = int(0.8 * len(full_length_dataset))
    test_size = len(full_length_dataset) - train_size

    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(
        full_length_dataset, [train_size, test_size]
    )

    # dataset = FixedLengthChordDataset(test_dataset, segment_length=10)

    # Initialize the model architecture
    model = CRNN(
        input_features=test_dataset.dataset.n_bins, num_classes=NUM_CHORDS, cr2=False
    )

    # Load the trained weights
    exp_name = "large-vocab-fewer-X"
    save_path = os.path.join(f"data/experiments/{exp_name}/best_model.pth")
    model.load_state_dict(torch.load(save_path, weights_only=True))

    metrics = evaluate_model(model, test_dataset, batch_size=32)
    print(metrics)


if __name__ == "__main__":
    main()
