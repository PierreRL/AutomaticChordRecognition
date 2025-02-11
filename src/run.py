import autorootcwd
import os
import argparse
from datetime import datetime
import torch

from src.train import train_model, TrainingArgs
from src.data.dataset import (
    FixedLengthRandomChordDataset,
    FixedLengthChordDataset,
    FullChordDataset,
)
from src.models.ismir2017 import ISMIR2017ACR
from src.utils import (
    NUM_CHORDS,
    HOP_LENGTH,
    write_json,
    write_text,
    get_filenames,
    get_split_filenames,
)
from src.eval import evaluate_model


def main():
    # Set up CLI arguments
    parser = argparse.ArgumentParser(
        description="Train and evaluate a chord recognition model."
    )
    parser.add_argument(
        "--exp_name", type=str, help="Name of the experiment.", required=True
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs to train."
    )
    parser.add_argument(
        "--decrease_lr_epochs",
        type=int,
        default=10,
        help="Number of epochs before decreasing LR.",
    )
    parser.add_argument(
        "--decrease_lr_factor",
        type=float,
        default=0.5,
        help="Factor by which to decrease LR.",
    )
    parser.add_argument(
        "--early_stopping", type=int, default=20, help="Early stopping patience."
    )
    parser.add_argument(
        "--random_pitch_shift",
        action="store_true",
        help="Whether to apply random pitch shift.",
    )
    parser.add_argument(
        "--cr2",
        action="store_true",
        help="Whether to use the cr2 version of ISMIR2017.",
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Whether to use cached CQT and chord annotation files.",
    )
    # TODO:
    parser.add_argument(
        "--hop_length",
        type=int,
        default=4096,
        help="Hop length used to compute the log CQT.",
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        default=10,
        help="Segment length for training dataset in seconds.",
    )
    parser.add_argument(
        "--weighted_loss",
        action="store_true",
        help="Whether to use weighted loss.",
    )
    parser.add_argument(
        "--train_on_X",
        action="store_true",
        help="Whether to train on class lable X.",
    )

    args = parser.parse_args()

    torch.manual_seed(0)

    print("=" * 50)
    print(f"Running experiment: {args.exp_name}")
    print("=" * 50)

    # Create a directory to store the experiment results
    DIR = f"./data/experiments/{args.exp_name}"
    os.makedirs(DIR)

    # Load the dataset
    all_filenames = get_filenames()

    # Split the dataset into train, validation, and test
    train_size = int(0.8 * len(all_filenames))
    val_size = int(0.25 * train_size)
    test_size = len(all_filenames) - train_size
    train_filenames, val_filenames, test_filenames = get_split_filenames()

    # Create datasets
    train_dataset = FixedLengthRandomChordDataset(
        filenames=train_filenames,
        random_pitch_shift=args.random_pitch_shift,
        cached=args.cached,
    )
    val_dataset = FixedLengthChordDataset(
        filenames=val_filenames, segment_length=args.segment_length, cached=args.cached
    )
    test_dataset = FullChordDataset(filenames=test_filenames, cached=args.cached)

    # Initialize the model
    model = ISMIR2017ACR(
        input_features=train_dataset.full_dataset.n_bins,
        num_classes=NUM_CHORDS,
        cr2=args.cr2,
    )

    # Save the experiment name and time
    run_metadata = {
        "experiment_name": args.exp_name,
        "time": str(datetime.now()),
        "model": str(model),
        "dataset": {
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "NUM_CHORDS": NUM_CHORDS,
        },
    }
    write_json(run_metadata, f"{DIR}/metadata.json")

    training_args = TrainingArgs(
        num_epochs=args.num_epochs,
        lr=args.lr,
        segment_length=args.segment_length,
        decrease_lr_epochs=args.decrease_lr_epochs,
        decrease_lr_factor=args.decrease_lr_factor,
        early_stopping=args.early_stopping,
        do_validation=True,
        save_model=True,
        save_dir=f"{DIR}/",
        save_filename="best_model.pth",
        device=None,
    )

    # Save the training args
    write_json(training_args._asdict(), f"{DIR}/training_args.json")

    # Train the model
    print(f"Number of training samples: {len(train_dataset)}")
    print("Training model...")
    try:
        training_history = train_model(
            model,
            train_dataset,
            val_dataset,
            args=training_args,
        )
    except Exception as e:
        print(f"Training Error: {e}")
        # Save the exception to a file
        write_text(f"{DIR}/training_error.txt", str(e))

    # Evaluate the model
    try:
        model.load_state_dict(torch.load(f"{DIR}/best_model.pth"))
        metrics = evaluate_model(model, test_dataset)
    except Exception as e:
        print(f"Evaluation Error: {e}")
        # Save the exception to a file
        write_text(str(e), f"{DIR}/evaluation_error.txt")

    # Save the training history dictionary and metrics dictionary as json files
    write_json(training_history, f"{DIR}/training_history.json")
    write_json(metrics, f"{DIR}/metrics.json")


if __name__ == "__main__":
    main()
