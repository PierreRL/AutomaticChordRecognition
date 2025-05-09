import autorootcwd
import os
import argparse
from datetime import datetime
import torch

from src.train import TrainingArgs, train_model
from src.eval import evaluate_model
from src.data.dataset import generate_datasets, get_calibrated_priors
from src.models.crnn import CRNN
from src.models.cnn import CNN
from src.models.logistic_acr import LogisticACR
from src.utils import (
    write_json,
    generate_experiment_name,
    get_torch_device,
    NUM_CHORDS,
    N_BINS,
    N_MELS,
    N_FFT,
)


def main():
    start_time = datetime.now()

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
        default="crnn",
        help="Model to train. Values: crnn, logistic.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=64, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--early_stopping", type=int, default=25, help="Early stopping patience."
    )
    parser.add_argument(
        "--enable_early_stopping", action="store_true", help="Enable early stopping."
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
        "--spectrogram_type",
        type=str,
        default="cqt",
        help="Type of spectrogram to use. Values: cqt, linear, mel, chroma",
    )
    parser.add_argument(
        "--cqt_pitch_shift",
        action="store_true",
        help="Whether to apply random pitch shift directly to CQT features.",
    )
    parser.add_argument(
        "--audio_pitch_shift",
        action="store_true",
        help="Whether to apply random pitch shift directly to audio.",
    )
    parser.add_argument(
        "--aug_shift_prob",
        type=float,
        default=0.9,
        help="Probability of applying pitch shift augmentation.",
    )
    parser.add_argument(
        "--use_generative_features",
        action="store_true",
        help="Whether to use generative features.",
    )
    parser.add_argument(
        "--no_cqt",
        dest="use_cqt",
        action="store_false",
        help="Whether to use CQT features.",
    )
    parser.add_argument(
        "--cr2",
        action="store_true",
        help="Whether to use the cr2 version of crnn, with comparable model size.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Hidden size of the GRU layers.",
    )
    parser.add_argument(
        "--cnn_layers",
        type=int,
        default=1,
        help="Number of layers in the CNN model.",
    )
    parser.add_argument(
        "--gru_layers",
        type=int,
        default=1,
        help="Number of layers in the GRU for the CRNN model.",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=None,
        help="Hop length used to compute the log CQT.",
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        default=10,
        help="Segment length for training dataset in seconds.",
    )
    parser.add_argument(
        "--cnn_channels",
        type=int,
        default=1,
        help="Number of channels in the CNN model.",
    )
    parser.add_argument(
        "--cnn_kernel_size",
        type=int,
        default=5,
        help="Kernel size for the CNN model.",
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
        "--weight_alpha",
        type=float,
        default=0.3,
        help="Alpha smoothing parameter for the weighted loss.",
    )
    parser.add_argument(
        "--input_transitions",
        action="store_true",
        help="Whether to pass transitions to the model. Should not be used with beat-wise resampling.",
    )
    parser.add_argument(
        "--structured_loss",
        action="store_true",
        help="Use structured loss.",
    )
    parser.add_argument(
        "--structured_loss_alpha",
        type=float,
        default=0.7,
        help="Alpha parameter for the structured loss.",
    )
    parser.add_argument("--crf", action="store_true", help="Use CRF module.")
    parser.add_argument(
        "--hmm_smoothing",
        action="store_true",
        help="Use HMM smoothing.",
    )
    parser.add_argument(
        "--hmm_alpha",
        type=float,
        default=0.2,
        help="Alpha parameter for the HMM smoothing. The probability of staying in the same chord.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="60",
        help="Train split percentage. Must be 60, 80, or 100.",
    )
    parser.add_argument(
        "--generative_features_dim",
        type=int,
        default=2048,
        help="Dimensionality of the generative feature codebooks. If gen_reduction is concat, this is multiplied by 4. assuming 4 codebooks.",
    )
    parser.add_argument(
        "--gen_down_dimension",
        type=int,
        default=64,
        help="Dimensionality of the generative feature vector after projection.",
    )
    parser.add_argument(
        "--gen_reduction",
        type=str,
        default="codebook_3",
        help="Reduction method for generative features. Values: avg, concat, codebook_0, codebook_1, codebook_2, codebook_3.",
    )
    parser.add_argument(
        "--gen_model_name",
        type=str,
        default="large",
        help="Size of the generative model. Values: small, large, chord, melody.",
    )
    parser.add_argument(
        "--beat_wise_resample",
        action="store_true",
        help="Whether to resample the input features to the beat level.",
    )
    parser.add_argument(
        "--beat_resample_interval",
        type=float,
        default=1,
        help="Interval in seconds for beat-wise resampling.",
    )
    parser.add_argument(
        "--perfect_beat_resample",
        action="store_true",
        help="Whether to use perfect beat resampling (uses labels).",
    )
    parser.add_argument(
        "--perfect_beat_resample_eval",
        action="store_true",
        help="Whether to use perfect beat resampling for evaluation. This is an unfair metric since it uses the labels.",
    )
    parser.add_argument(
        "--job_id",
        type=str,
        default=None,
        help="Job ID for the experiment. Used for tracking in the cluster.",
    )
    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="Whether to use synthetic data.",
    )
    parser.add_argument(
        "--synthetic_ratio",
        type=float,
        default=1,
        help="Ratio of synthetic data to real data.",
    )
    parser.add_argument(
        "--synthetic_input_dir",
        type=str,
        default="./data/synthetic",
        help="Directory containing the synthetic data.",
    )
    parser.add_argument(
        "--synthetic_only",
        action="store_true",
        help="Whether to use only synthetic data for training.",
    )
    parser.add_argument(
        "--test_on_synthetic",
        action="store_true",
        help="Whether to test on synthetic data.",
    )
    parser.add_argument(
        "--no_calibrate_logits",
        action="store_false",
        dest="calibrate_logits",
        help="Disable calibration of logits to synthetic data.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility.")
    parser.add_argument(
        "--fdr",  # Fast Debug Run for faster testing. Sets datasets to size 10 and epoch 1.
        action="store_true",
        help="Run a single batch of training and validation and small evaluation set.",
    )
    parser.add_argument("--no_train", action="store_true", help="Skip training.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if (
        args.hop_length is None
    ):  # Default hop length is 4096, but if beat-wise resampling is used, it is set to 1024.
        if args.beat_wise_resample:
            args.hop_length = 1024
        else:
            args.hop_length = 4096

    assert not (args.use_synthetic and args.weight_loss), (
        "Synthetic data is not supported with weighted loss."
    )

    assert not (args.use_synthetic and args.use_generative_features), (
        "Generative features are not supported with synthetic data."
    )
    assert not (args.use_generative_features and args.cqt_pitch_shift), (
        "CQT pitch shift is not supported with generative features."
    )

    assert args.train_split in [
        "60",
        "80",
        "100",
    ], "train_split must be 60, 80, or 100."
    assert args.spectrogram_type in [
        "cqt",
        "linear",
        "mel",
        "chroma",
    ], "spectrogram_type must be cqt, linear, mel, or chroma."
    assert args.model in [
        "crnn",
        "logistic",
        "cnn",
    ], "model must be crnn, logistic, or cnn."
    assert args.gen_reduction in [
        "avg",
        "concat",
        "codebook_0",
        "codebook_1",
        "codebook_2",
        "codebook_3",
    ], "gen_reduction must be avg, concat, codebook_0, codebook_1, codebook_2, or codebook_3."
    assert args.optimiser in ["adam", "sgd"], "optimiser must be adam or sgd."
    assert args.lr_scheduler in [
        "cosine",
        "plateau",
        "none",
    ], "lr_scheduler must be cosine, plateau, or none."

    # Cannot have CRF and HMM
    if args.crf and args.hmm_smoothing:
        raise ValueError("Cannot use both CRF and HMM smoothing.")

    # Cannot have structured loss and CRF
    if args.structured_loss and args.crf:
        raise ValueError("Cannot use both structured loss and CRF.")
    
    if args.use_synthetic:
        assert args.synthetic_input_dir is not None, (
            "Synthetic input directory must be specified when using synthetic data."
        )
    
    if args.test_on_synthetic:
        assert args.synthetic_input_dir is not None, (
            "Synthetic input directory must be specified when testing on synthetic data."
        )

    if not args.exp_name:
        args.exp_name = generate_experiment_name()
    
    use_augs = args.audio_pitch_shift or args.cqt_pitch_shift

    print("=" * 50)
    print(f"Running experiment: {args.exp_name}")
    print("=" * 50)

    # Create a directory to store the experiment results
    DIR = f"{args.output_dir}/{args.exp_name}"
    os.makedirs(DIR, exist_ok=True)

    # Create datasets
    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_final_test_dataset,
        val_final_test_dataset,
        synthetic_final_test_dataset,
    ) = generate_datasets(
        train_split=args.train_split,
        input_dir=args.input_dir,
        segment_length=args.segment_length,
        cqt_pitch_shift=args.cqt_pitch_shift,
        audio_pitch_shift=args.audio_pitch_shift,
        aug_shift_prob=args.aug_shift_prob,
        hop_length=args.hop_length,
        input_transitions=args.input_transitions,
        mask_X=args.mask_X,
        gen_reduction=args.gen_reduction if args.use_generative_features else None,
        gen_model_name=args.gen_model_name if args.use_generative_features else None,
        spectrogram_type=args.spectrogram_type,
        beat_wise_resample=args.beat_wise_resample,
        beat_resample_interval=args.beat_resample_interval,
        perfect_beat_resample=args.perfect_beat_resample,
        perfect_beat_resample_eval=args.perfect_beat_resample_eval,
        subset_size=(10 if args.fdr else None),  # We subset for FDR
        synthetic_ratio=args.synthetic_ratio if args.use_synthetic else None,
        synthetic_input_dir=args.synthetic_input_dir,
        use_synthetic=args.use_synthetic,
        synthetic_only=args.synthetic_only,
        test_on_synthetic=args.test_on_synthetic
    )

    # Params for Fast Development Run (FDR)
    if args.fdr:
        args.validate_every = 1
        args.epochs = 1
        args.output_dir = "experiments_fdr"

    input_features_mapping = {
        "cqt": N_BINS,
        "linear": N_FFT // 2 + 1,
        "mel": N_MELS,
        "chroma": 12,
    }
    input_features = input_features_mapping.get(args.spectrogram_type, N_BINS)

    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    # Initialize the model
    if args.model == "crnn":
        model = CRNN(
            input_features=input_features,
            num_classes=NUM_CHORDS,
            cr2=args.cr2,
            hidden_size=args.hidden_size,
            gru_layers=args.gru_layers,
            cnn_layers=args.cnn_layers,
            cnn_channels=args.cnn_channels,
            kernel_size=args.cnn_kernel_size,
            hmm_smoothing=args.hmm_smoothing,
            hmm_alpha=args.hmm_alpha,
            use_cqt=args.use_cqt,
            input_transitions=args.input_transitions,
            use_generative_features=args.use_generative_features,
            gen_down_dimension=args.gen_down_dimension,
            gen_dimension=(
                4 * args.generative_features_dim
                if args.gen_reduction == "concat"
                else args.generative_features_dim
            ),  # Concat 4 codebooks, all other reductions are 1 codebook
            structured_loss=args.structured_loss,
            crf=args.crf,
        )
    elif args.model == "logistic":
        model = LogisticACR(
            input_features=input_features,
            num_classes=NUM_CHORDS,
            hmm_smoothing=args.hmm_smoothing,
            hmm_alpha=args.hmm_alpha,
        )
    elif args.model == "cnn":
        model = CNN(
            input_features=input_features,
            num_classes=NUM_CHORDS,
            hmm_smoothing=args.hmm_smoothing,
            hmm_alpha=args.hmm_alpha,
            use_cqt=args.use_cqt,
            use_generative_features=args.use_generative_features,
            gen_dimension=args.generative_features_dim,
            num_layers=args.cnn_layers,
            kernel_size=args.cnn_kernel_size,
            channels=args.cnn_channels,
            activation="relu",
        )
    elif args.model == "transformer":
        raise NotImplementedError("Transformer model not implemented yet.")
    else:
        raise ValueError(
            f"Invalid model type {args.model}. Must be one of crnn, logistic, transformer."
        )

    # Save the experiment name and time
    run_metadata = {
        "experiment_name": args.exp_name,
        "start_time": str(start_time),
        "model": model.to_dict(),
        "seed": args.seed,
        "job_id": args.job_id,
        "dataset": {
            "train_size": len(train_final_test_dataset),
            "val_size": len(val_final_test_dataset),
            "test_size": len(test_dataset),
            "NUM_CHORDS": NUM_CHORDS,
        },
        "hmm_smoothing": args.hmm_smoothing,
        "hmm_alpha": args.hmm_alpha,
        "use_cqt": args.use_cqt,
        "use_generative_features": args.use_generative_features,
        "args": vars(args),
    }
    write_json(run_metadata, f"{DIR}/metadata.json")


    if not args.no_train:
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
            structured_loss=args.structured_loss,
            structured_loss_alpha=args.structured_loss_alpha,
            use_weighted_loss=args.weight_loss,
            weight_alpha=args.weight_alpha,
            weight_decay=args.weight_decay,
            aug_shift_prob=args.aug_shift_prob if use_augs else 0,
            use_augs=use_augs,
            lr_scheduler=args.lr_scheduler,
            optimiser=args.optimiser,
            momentum=args.momentum,
            use_crf=args.crf,
            early_stopping=args.early_stopping if args.enable_early_stopping else None,
            do_validation=args.train_split == "60",
            save_dir=f"{DIR}",
            save_filename="best_model.pth",
        )

        # Save the training args
        write_json(training_args._asdict(), f"{DIR}/training_args.json")

        # Train the model
        print(f"Number of training samples: {len(train_dataset)}")
        print("Training model...")
        training_history = train_model(
            model,
            train_dataset,
            val_dataset,
            args=training_args,
        )

        # Save the training history dictionary
        write_json(training_history, f"{DIR}/training_history.json")

    # Validate and test the model

    # Load the best model
    model.load_state_dict(torch.load(f"{DIR}/best_model.pth", weights_only=True, map_location=get_torch_device()))
    model.eval()

    torch.set_grad_enabled(False)

    if args.calibrate_logits:
        log_calibration = get_calibrated_priors(
            train_dataset.full_dataset,
            train_final_test_dataset,
            aug_shift_prob=args.aug_shift_prob if use_augs else 0,
            root_invariance=True,
            return_as_log=True
        )
    else:
        log_calibration = None


    # Validation set
    if args.train_split == "60":
        print("What?")
        print("Evaluating model on validation set...")
        val_metrics = evaluate_model(
            model, 
            val_final_test_dataset, 
            batch_size=args.eval_batch_size, 
            log_calibration=log_calibration
        )
        write_json(val_metrics, f"{DIR}/val_metrics.json")

    # Test set
    if args.train_split != "100":
        print("Evaluating model on test...")
        test_metrics = evaluate_model(
            model, 
            test_dataset, 
            batch_size=args.eval_batch_size, 
            log_calibration=log_calibration
        )
        write_json(test_metrics, f"{DIR}/test_metrics.json")

    # Train set
    print("Evaluating model on train...")
    train_metrics = evaluate_model(
        model, 
        train_final_test_dataset, 
        batch_size=args.eval_batch_size, 
        log_calibration=log_calibration
    )
    write_json(train_metrics, f"{DIR}/train_metrics.json")

    # Test on synthetic train set
    if args.test_on_synthetic:
        print("Evaluating model on synthetic test set...")
        synthetic_metrics = evaluate_model(
            model, synthetic_final_test_dataset, batch_size=args.eval_batch_size
        )
        write_json(synthetic_metrics, f"{DIR}/synthetic_metrics.json")

    # Calculate elapsed time
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_time = str(elapsed_time).split(",")[0]
    print(f"Elapsed time: {elapsed_time}")
    run_metadata["elapsed_time"] = str(elapsed_time)
    write_json(run_metadata, f"{DIR}/metadata.json")

    print("=" * 50)
    print(f"Experiment {args.exp_name} completed.")
    print("=" * 50)


if __name__ == "__main__":
    main()
