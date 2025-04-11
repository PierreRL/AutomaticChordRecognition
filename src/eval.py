import autorootcwd
from tqdm import tqdm
from enum import Enum
import os
from typing import List, Dict

import torch
import numpy as np
from torch.utils.data import DataLoader
import mir_eval

from src.models.crnn import CRNN
from src.models.base_model import BaseACR
from src.data.dataset import FullChordDataset, IndexedDataset
from src.data.beats.beatwise_resampling import get_resampled_full_beats
from src.utils import (
    get_torch_device,
    collate_fn_indexed,
    get_chord_seq,
    id_to_chord_table,
    chord_to_id_map,
    NUM_CHORDS,
    SMALL_VOCABULARY,
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
    TRIADS = "triads"
    TETRADS = "tetrads"

    def eval_func(self) -> callable:
        """
        Returns the evaluation function for this evaluation metric.
        """
        if self == EvalMetric.ROOT:
            return mir_eval.chord.root
        elif self == EvalMetric.MIREX:
            return mir_eval.chord.mirex
        elif self == EvalMetric.THIRD:
            return mir_eval.chord.thirds
        elif self == EvalMetric.SEVENTH:
            return mir_eval.chord.sevenths
        elif self == EvalMetric.MAJMIN:
            return mir_eval.chord.majmin
        elif self == EvalMetric.TRIADS:
            return mir_eval.chord.triads
        elif self == EvalMetric.TETRADS:
            return mir_eval.chord.tetrads
        elif self == EvalMetric.SONG_WISE_ACC:

            def song_wise_acc(hypotheses: torch.Tensor, references: torch.Tensor):
                return np.array([h == r for h, r in zip(hypotheses, references)])

            return song_wise_acc
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

        return metrics * 100  # Convert to percentage


def evaluate_model(
    model: BaseACR,
    dataset: FullChordDataset,
    evals: List[EvalMetric] = [
        EvalMetric.ROOT,
        EvalMetric.MIREX,
        EvalMetric.THIRD,
        EvalMetric.SEVENTH,
        EvalMetric.MAJMIN,
        EvalMetric.TRIADS,
        EvalMetric.TETRADS,
        EvalMetric.SONG_WISE_ACC,
    ],
    batch_size: int = 32,
    device: torch.device = None,
) -> dict:
    """
    Evaluate a model using continuous, song-based metrics computed with mir_eval.

    This function first iterates over the dataset DataLoader to collect a list of per-song
    predictions and ground truth chord sequences. Then, for each song, it:

      - Retrieves the full predicted and reference chord sequences (predictions and ground truth)
      - Gets the song’s beat boundaries (via dataset.get_beats(idx))
      - Forms beat intervals as [b[i], b[i+1]] (assumed to correspond to each prediction)
      - Adjusts and merges intervals with mir_eval.util.adjust_intervals and merge_labeled_intervals
      - Computes durations from the intervals and then uses mir_eval.chord.weighted_accuracy
        with the chosen chord-comparison function (e.g. mir_eval.chord.thirds)
      - Computes the number of transitions in the predicted sequence

    Finally, the function returns a dictionary containing the mean continuous score per evaluation
    metric and the overall average number of transitions per song.

    Note: This implementation ignores X chords (and any invalid labels, e.g. -1) when converting
    IDs to chord strings.

    Args:
        model (BaseACR): The trained chord recognition model.
        dataset: A dataset instance that returns full-song features and chord labels and provides a get_beats() method.
        evals (list): A list of strings indicating which mir_eval chord evaluation function to use.
                      For example: ["thirds", "majmin", "root", "mirex", "sevenths"].
        batch_size (int): The batch size to use for iterating over the dataset.
        device (torch.device, optional): The device on which to run evaluation.

    Returns:
        results (dict): A dictionary with the following keys:
            - For each metric in evals, a mean score (over songs) computed by mir_eval.chord.weighted_accuracy.
            - "avg_transitions": the mean number of chord transitions per song.
    """
    torch.set_grad_enabled(False)
    if device is None:
        device = get_torch_device()
    model.to(device)
    model.eval()

    filenamed_dataset = IndexedDataset(dataset)

    data_loader = DataLoader(
        filenamed_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_indexed,
    )

    print("Evaluating model...")
    song_predictions = []

    for (batch_cqts, batch_gens, batch_labels), indices in tqdm(
        data_loader, desc="Predicting"
    ):
        batch_cqts = batch_cqts.to(device)
        if batch_gens is not None and batch_gens.nelement() > 0:
            batch_gens = batch_gens.to(device)
        batch_labels = batch_labels.to(device)

        if SMALL_VOCABULARY:
            valid_mask = batch_labels != -1
        else:
            valid_mask = torch.logical_and(
                batch_labels != -1, batch_labels != chord_to_id_map["X"]
            )

        ignore_mask = batch_labels != -1

        if hasattr(model, "use_generative_features") and model.use_generative_features:
            predictions = model.predict(
                batch_cqts, batch_gens, mask=valid_mask, device=device
            )
        else:
            predictions = model.predict(batch_cqts, mask=valid_mask)

        predictions = predictions.cpu().numpy()

        for i in range(predictions.shape[0]):
            song_predictions.append(
                {
                    "pred_ids": predictions[i][ignore_mask[i]].tolist(),
                    "idx": indices[i],
                }
            )

    song_metric_scores = {m: [] for m in evals}
    song_transition_counts = []
    song_agg_data = []

    for song in tqdm(song_predictions, desc="Evaluating"):
        filename = dataset.get_filename(song["idx"])
        pred_labels = [id_to_chord_table[x] for x in song["pred_ids"]]

        # Get estimated beat boundaries (from the features) and reference beat boundaries.
        est_beats = dataset.get_beats(song["idx"])
        ref_beats = get_resampled_full_beats(filename, perfect_beat_resample=True)

        # Get ground-truth chord sequence (one label per reference beat interval).
        ref_labels = get_chord_seq(filename)

        # Convert beat boundaries into intervals.
        est_intervals = np.column_stack((est_beats[:-1], est_beats[1:]))
        ref_intervals = np.column_stack((ref_beats[:-1], ref_beats[1:]))

        # Adjust the estimated intervals so that they span the same range as the reference intervals.
        adjusted_est_intervals, est_labels = mir_eval.util.adjust_intervals(
            est_intervals,
            pred_labels,
            ref_intervals.min(),
            ref_intervals.max(),
            mir_eval.chord.NO_CHORD,
            mir_eval.chord.NO_CHORD,
        )

        merged_intervals, merged_ref, merged_est = (
            mir_eval.util.merge_labeled_intervals(
                ref_intervals, ref_labels, adjusted_est_intervals, est_labels
            )
        )
        durations = mir_eval.util.intervals_to_durations(merged_intervals)

        merged_ref = np.array(merged_ref)
        merged_est = np.array(merged_est)
        durations = np.array(durations)

        # Mask out X chords
        mask_no_X = merged_ref != "X"
        merged_ref = merged_ref[mask_no_X]
        merged_est = merged_est[mask_no_X]
        durations = durations[mask_no_X]

        # Save aggregated data for class-wise metrics.
        song_agg_data.append(
            {"merged_ref": merged_ref, "merged_est": merged_est, "durations": durations}
        )

        for e in evals:
            comp = e.evaluate(hypotheses=merged_est, references=merged_ref)
            score = mir_eval.chord.weighted_accuracy(comp, durations)
            song_metric_scores[e].append(score)

        # Compute number of transitions in the predicted sequence.
        pred_transitions = sum(
            1
            for j in range(len(pred_labels) - 1)
            if pred_labels[j] != pred_labels[j + 1]
        )
        song_transition_counts.append(pred_transitions)

    results = {}
    results["mean"] = {}
    results["median"] = {}
    results["std"] = {}
    results["boostrap-stde"] = {}
    results["bootstrap-95ci"] = {}
    for m in evals:
        results["mean"][m.value] = np.mean(song_metric_scores[m])
        results["median"][m.value] = np.median(song_metric_scores[m])
        results["std"][m.value] = np.std(song_metric_scores[m])
        _, se, ci = bootstrap_mean_ci(song_metric_scores[m], num_bootstrap=10000, ci=95)
        results["boostrap-stde"][m.value] = se
        results["bootstrap-95ci"][m.value] = ci

    results["avg_transitions_per_song"] = np.mean(song_transition_counts)

    class_agg_results = {}
    for e in tqdm(evals, desc="Class-wise metrics"):
        # Compute the overall (aggregated) metric using mean and median over chords.
        aggregated_class_mean = compute_aggregated_class_metric(
            song_agg_data, e, np.mean
        )
        aggregated_class_median = compute_aggregated_class_metric(
            song_agg_data, e, np.median
        )
        class_agg_results[m.value] = {
            "mean": aggregated_class_mean,
            "median": aggregated_class_median,
        }
    results["class_wise"] = class_agg_results

    return results


def bootstrap_mean_ci(data, num_bootstrap=10000, ci=95):
    """
    Bootstrap the mean of a dataset and calculate confidence intervals.
    Args:
        data (array-like): The data to bootstrap.
        num_bootstrap (int): The number of bootstrap samples.
        ci (float): The confidence interval to calculate.

    Returns:
        mean (float): The mean of the bootstrap samples.
        se (float): The standard error of the mean.
        ci (tuple): The lower and upper bounds of the confidence interval. Should be in the range [50, 100].
    """
    boot_means = []
    n = len(data)
    for _ in range(num_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    # Standard error as the standard deviation of the bootstrapped means
    se = np.std(boot_means)
    # Calculate confidence intervals
    lower_bound = np.percentile(boot_means, (100 - ci) / 2)
    upper_bound = np.percentile(boot_means, 100 - (100 - ci) / 2)
    mean = np.mean(boot_means)
    return mean, se, (lower_bound, upper_bound)


def compute_aggregated_class_metric(
    song_agg_data: List[dict], eval_metric: EvalMetric, agg_func=np.mean
) -> float:
    """
    Compute an overall aggregated class metric.
    For each chord (as present in the aggregated data), compute the weighted accuracy
    using the specified evaluation function, then combine these chord-wise scores
    using an aggregation function (np.mean or np.median).

    song_agg_data: List of dicts (one per song) with keys "merged_ref", "merged_est", and "durations".
    eval_metric: An EvalMetric instance.
    agg_func: Aggregation function (np.mean or np.median).
    """
    # Aggregate intervals, labels, durations across all songs.
    agg_ref_all = np.concatenate([s["merged_ref"] for s in song_agg_data])
    agg_est_all = np.concatenate([s["merged_est"] for s in song_agg_data])
    agg_dur_all = np.concatenate([s["durations"] for s in song_agg_data])
    unique_chords = np.unique(agg_ref_all)
    chord_scores = []
    for chord in unique_chords:
        if chord == "X":
            continue  # Skip 'X' chord
        mask = agg_ref_all == chord
        if np.sum(mask) == 0:
            continue
        comp = eval_metric.evaluate(agg_est_all[mask], agg_ref_all[mask])
        score = mir_eval.chord.weighted_accuracy(comp, agg_dur_all[mask])
        chord_scores.append(score)
    if len(chord_scores) == 0:
        return np.nan
    return agg_func(chord_scores)


"""
Legacy code for evaluating the model using discrete metrics. Now updated to use continuous metrics - 'weighted recall score'.
"""
# def evaluate_model_old_discrete(
#     model: BaseACR,
#     dataset: FullChordDataset,
#     evals: List[EvalMetric] = [
#         EvalMetric.ROOT,
#         EvalMetric.MAJMIN,
#         EvalMetric.MIREX,
#         EvalMetric.THIRD,
#         EvalMetric.SEVENTH,
#         EvalMetric.SONG_WISE_ACC,
#     ],
#     batch_size: int = 64,
#     device: torch.device = None,
# ) -> Dict[str, float]:
#     """
#     Evaluate a model on a dataset split using a list of evaluation metrics.

#     Args:
#         model (BaseACRModel): The model to evaluate.
#         dataset (FullChordDataset): The dataset to evaluate on.
#         evals (list[EvalMetrics]): The evaluation metrics to use. Defaults to [EvalMetrics.ROOT, EvalMetrics.MAJMIN, EvalMetrics.MIREX, EvalMetrics.THIRD, EvalMetrics.SEVENTH].
#         batch_size (int): The batch size to use for evaluation. Defaults to 32.
#         device (torch.device): The device to use for evaluation. Defaults to None.

#     Returns:
#         metrics (dict[str, float]): A dictionary of evaluation metrics and their values.
#     """
#     torch.set_grad_enabled(False)  # Disable gradient calculation for evaluation
#     if not device:
#         device = get_torch_device()

#     model.to(device)
#     model.eval()

#     # Initialize metrics storage
#     metrics = {"mean": {}, "median": {}}

#     for eval in evals:
#         metrics["mean"][eval.value] = 0.0
#         metrics["median"][eval.value] = []

#     data_loader = DataLoader(
#         dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
#     )

#     song_metrics = {eval.value: {} for eval in evals}

#     all_hypotheses = []
#     all_references = []

#     for batch_cqts, batch_gens, batch_labels in tqdm(data_loader):
#         batch_cqts, batch_gens, batch_labels = (
#             batch_cqts.to(device),
#             batch_gens.to(device),
#             batch_labels.to(device),
#         )

#         # Compute valid frame mask
#         if SMALL_VOCABULARY:
#             valid_mask = batch_labels != -1
#         else:
#             valid_mask = torch.logical_and(
#                 batch_labels != -1, batch_labels != chord_to_id_map["X"]
#             )

#         if hasattr(model, "use_generative_features") and model.use_generative_features:
#             predictions = model.predict(
#                 batch_cqts, batch_gens, mask=valid_mask, device=device
#             ).to(device)
#         else:
#             predictions = model.predict(batch_cqts, mask=valid_mask).to(device)

#         for i in range(batch_labels.shape[0]):  # Iterate over songs in the batch

#             filtered_references = batch_labels[i][valid_mask[i]].cpu().numpy()
#             filtered_hypotheses = predictions[i][valid_mask[i]].cpu().numpy()

#             all_hypotheses.extend(filtered_hypotheses)
#             all_references.extend(filtered_references)

#             ref_labels = id_to_chord_table[filtered_references]
#             hyp_labels = id_to_chord_table[filtered_hypotheses]

#             for eval in evals:
#                 if eval.get_eval_input_type() == "int":
#                     song_eval_scores = eval.evaluate(
#                         filtered_references, filtered_hypotheses
#                     )
#                 else:
#                     song_eval_scores = eval.evaluate(hyp_labels, ref_labels)

#                 # Filter out invalid scores that are -1 (produced by mir_eval with 'X' labels for example)
#                 song_eval_scores = song_eval_scores[song_eval_scores != -1]

#                 if i not in song_metrics[eval.value]:
#                     song_metrics[eval.value][i] = []

#                 song_metrics[eval.value][i].append(np.mean(song_eval_scores))

#     for eval in evals:
#         song_scores = [np.mean(scores) for scores in song_metrics[eval.value].values()]

#         metrics["mean"][eval.value] = np.mean(song_scores)
#         metrics["median"][eval.value] = np.median(song_scores)

#     # Flatten along song dimension
#     all_hypotheses = np.array(all_hypotheses)
#     all_references = np.array(all_references)

#     # Compute frame-wise accuracy
#     metrics["frame_wise_acc"] = accuracy_score(all_references, all_hypotheses)

#     # Compute class-wise accuracy
#     class_accs = np.full(NUM_CHORDS, np.nan)
#     for i in range(NUM_CHORDS):
#         # Find all references of class i
#         mask = all_references == i
#         class_references = all_references[mask]
#         class_hypotheses = all_hypotheses[mask]

#         if class_references.size > 0:
#             class_accs[i] = accuracy_score(class_references, class_hypotheses)

#         metrics["class_wise_acc_mean"] = np.nanmean(class_accs)  # Ignore NaNs
#         metrics["class_wise_acc_median"] = np.nanmedian(class_accs)

#     return metrics


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
