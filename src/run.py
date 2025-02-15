import autorootcwd
import os
import argparse
from datetime import datetime
import torch

from src.train import train_model, TrainingArgs
from src.data.dataset import generate_datasets
from src.models.ismir2017 import ISMIR2017ACR
from src.models.logistic_regressor import LogisticACR
from src.utils import (
    NUM_CHORDS,
    N_BINS,
    write_json,
    write_text,
    get_split_filenames,
    generate_experiment_name,
)
from src.eval import evaluate_model


def main():
    # Set up CLI arguments
    parser = argparse.ArgumentParser(
        description="Train and evaluate a chord recognition model."
    )
    parser.add_argument("--exp_name", type=str, help="Name of the experiment.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/processed",
        help="Directory containing the processed data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="Directory to store the experiment results.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ismir2017",
        help="Model to train. Values: ismir2017, logistic, transformer.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--early_stopping", type=int, default=40, help="Early stopping patience."
    )
    parser.add_argument(
        "--no_early_stopping", action="store_true", help="Disable early stopping."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="Weight decay for optimizer."
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="Learning rate scheduler. Values: cosine, plateau, none.",
    )
    parser.add_argument(
        "--validate_every",
        type=int,
        default=5,
        help="Frequency of validation evaluation in epochs.",
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
        "--optimiser",
        type=str,
        default="adam",
        help="Optimizer to use. Values: adam, sgd.",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for SGD optimizer."
    )
    parser.add_argument(
        "--random_pitch_shift",
        action="store_true",
        help="Whether to apply random pitch shift.",
    )
    parser.add_argument(
        "--cr2",
        action="store_true",
        help="Whether to use the cr2 version of ISMIR2017, with comparable model size.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Hidden size of the GRU layers.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of layers in the GRU.",
    )
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
        "--mask_X",
        action="store_true",
        help="Whether to ignore class label X for training.",
    )
    parser.add_argument(
        "--weight_loss",
        action="store_true",
        help="Whether to use weighted loss.",
    )
    parser.add_argument(
        "--fdr",  # Fast Debug Run for faster testing. Sets datasets to size 10 and epoch 1.
        action="store_true",
        help="Run a single batch of training and validation and small evaluation set.",
    )

    args = parser.parse_args()

    if not args.exp_name:
        args.exp_name = generate_experiment_name()

    torch.manual_seed(0)

    print("=" * 50)
    print(f"Running experiment: {args.exp_name}")
    print("=" * 50)

    # Create a directory to store the experiment results
    DIR = f"{args.output_dir}/{args.exp_name}"
    os.makedirs(DIR)

    # Load the dataset filenames
    train_filenames, val_filenames, test_filenames = get_split_filenames()

    # Create datasets
    train_dataset, val_dataset, test_dataset, val_final_test_dataset = (
        generate_datasets(
            train_filenames,
            val_filenames,
            test_filenames,
            input_dir=args.input_dir,
            segment_length=args.segment_length,
            random_pitch_shift=args.random_pitch_shift,
            hop_length=args.hop_length,
            mask_X=args.mask_X,
            subset_size=(10 if args.fdr else None),  # We subset for FDR
        )
    )

    # Params for Fast Development Run (FDR)
    if args.fdr:
        args.validate_every = 1
        args.epochs = 1
        args.output_dir = "experiments_fdr"

    # Initialize the model
    if args.model == "ismir2017":
        model = ISMIR2017ACR(
            input_features=N_BINS,
            num_classes=NUM_CHORDS,
            cr2=args.cr2,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )
    elif args.model == "logistic":
        model = LogisticACR(input_features=N_BINS, num_classes=NUM_CHORDS)
    elif args.model == "transformer":
        raise NotImplementedError("Transformer model not implemented yet.")
    else:
        raise ValueError(
            f"Invalid model type {args.model}. Must be one of ismir2017, logistic, transformer."
        )

    # Save the experiment name and time
    run_metadata = {
        "experiment_name": args.exp_name,
        "time": str(datetime.now()),
        "model": str(model),
        "dataset": {
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "NUM_CHORDS": NUM_CHORDS,
        },
    }
    write_json(run_metadata, f"{DIR}/metadata.json")

    training_args = TrainingArgs(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hop_length=args.hop_length,
        segment_length=args.segment_length,
        validate_every=args.validate_every,
        decrease_lr_epochs=args.decrease_lr_epochs,
        decrease_lr_factor=args.decrease_lr_factor,
        mask_X=args.mask_X,
        use_weighted_loss=args.weight_loss,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        optimiser=args.optimiser,
        momentum=args.momentum,
        early_stopping=args.early_stopping if not args.no_early_stopping else None,
        save_dir=f"{DIR}/",
        save_filename="best_model.pth",
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
        return

    # Save the training history dictionary
    write_json(training_history, f"{DIR}/training_history.json")

    # Evaluate the model on val and test
    try:
        # Load the best model
        model.load_state_dict(torch.load(f"{DIR}/best_model.pth", weights_only=True))

        # Validation set
        print("Evaluation model on validation set...")
        val_metrics = evaluate_model(model, val_final_test_dataset)
        write_json(val_metrics, f"{DIR}/val_metrics.json")

        # Test set
        print("Evaluating model on test...")
        test_metrics = evaluate_model(model, test_dataset)
        write_json(test_metrics, f"{DIR}/test_metrics.json")

    except Exception as e:
        print(f"Evaluation Error: {e}")
        # Save the exception to a file
        write_text(f"{DIR}/evaluation_error.txt", str(e))
        return

    print("=" * 50)
    print(f"Experiment {args.exp_name} completed.")
    print("=" * 50)


if __name__ == "__main__":
    main()
