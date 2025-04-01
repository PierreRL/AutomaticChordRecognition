"""
Training script for a model
"""

import autorootcwd
import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.crnn import CRNN
from src.data.dataset import FixedLengthRandomChordDataset, FixedLengthChordDataset
from src.utils import get_torch_device, collate_fn, write_json, NUM_CHORDS, N_BINS, EarlyStopper


class TrainingArgs:
    def __init__(
        self,
        epochs: int,
        lr: float,
        batch_size: int = 64,
        segment_length: int = 10,
        early_stopping: int = 50,
        decrease_lr_factor: float = 0.5,
        decrease_lr_epochs: int = 15,
        validate_every: int = 5,
        mask_X: bool = True,
        use_weighted_loss: bool = False,
        weight_alpha: float = 0.65,
        weight_decay: float = 0.0,
        optimiser: str = "adam",
        momentum: float = 0.9,
        lr_scheduler: str = "cosine",
        hop_length: int = 4096,
        save_dir: str = "data/models/",
        save_filename: str = "best_model.pth",
        device: torch.device = None,
    ):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.segment_length = segment_length
        self.validate_every = validate_every
        self.early_stopping = early_stopping
        self.decrease_lr_factor = decrease_lr_factor
        self.decrease_lr_epochs = decrease_lr_epochs
        self.mask_X = mask_X
        self.use_weighted_loss = use_weighted_loss
        self.weight_alpha = weight_alpha
        self.weight_decay = weight_decay
        self.optimiser = optimiser
        self.momentum = momentum
        self.lr_scheduler = lr_scheduler
        self.hop_length = hop_length
        self.save_dir = save_dir
        self.save_filename = save_filename
        self.device = device

    def _asdict(self):
        return {
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "segment_length": self.segment_length,
            "hop_length": self.hop_length,
            "early_stopping": self.early_stopping,
            "validate_every": self.validate_every,
            "mask_X": self.mask_X,
            "weight_loss": self.use_weighted_loss,
            "weight_alpha": self.weight_alpha,
            "weight_decay": self.weight_decay,
            "optimiser": self.optimiser,
            **({"momentum": self.momentum} if self.optimiser == "sgd" else {}),
            "lr_scheduler": self.lr_scheduler,
            "decrease_lr_factor": self.decrease_lr_factor,
            "decrease_lr_epochs": self.decrease_lr_epochs,
        }


def train_model(
    model: nn.Module,
    train_dataset: FixedLengthRandomChordDataset,
    val_dataset: FixedLengthChordDataset,
    args: TrainingArgs,
):
    """
    Train a model on the given data loader.

    Args:
        model (nn.Module): The model to train.
        train_loader (Dataset): The training dataset.
        val_loader (Dataset): The validation dataset.
        training_args (TrainingArgs): The training arguments.

    Returns:
        dict: A dictionary containing the training history.
    """

    torch.set_grad_enabled(True)

    if args.device is None:
        # Use GPU if available, check for cuda and mps
        device = get_torch_device()
        print("Using device:", device)

    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Loss function
    if args.use_weighted_loss:
        weights = train_dataset.full_dataset.get_class_weights(alpha=args.weight_alpha)
        weights = weights.to(device)
    else:
        weights = None
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=weights)

    # Optimiser
    if args.optimiser == "adam":
        optimiser = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimiser == "sgd":
        optimiser = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError("Invalid optimiser")

    # Learning rate scheduler
    if args.lr_scheduler == "cosine":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max=args.epochs,
            eta_min=args.lr / 10,
        )
    elif args.lr_scheduler == "plateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode="min",
            factor=args.decrease_lr_factor,
            patience=args.decrease_lr_epochs // args.validate_every,
        )
    elif args.lr_scheduler == "none":
        lr_scheduler = None
    else:
        raise ValueError("Invalid lr_scheduler")

    # Early stopping
    if args.early_stopping is not None:
        early_stopper = EarlyStopper(args.early_stopping // args.validate_every)

    val_losses = []
    val_accuracies = []
    train_losses = []
    learning_rates = [optimiser.param_groups[0]["lr"]]

    for epoch in tqdm(range(args.epochs)):
        torch.set_grad_enabled(True)
        model.train()
        train_loss = 0.0
        for i, (features, gens, labels) in enumerate(train_loader):
            features, gens, labels = features.to(device), gens.to(device), labels.to(device)
            optimiser.zero_grad()

            if hasattr(model, 'use_generative_features') and model.use_generative_features:
                outputs = model(features, gens)
            else:
                outputs = model(features)

            # Flatten the outputs and labels
            outputs = outputs.view(-1, outputs.shape[-1])  # (B*frames, num_classes)
            labels = labels.view(-1)  # (B*frames) 

            # Compute the loss and backpropagate
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        if lr_scheduler is not None and args.lr_scheduler != "plateau":
            lr_scheduler.step()

        # Validation every 5 epochs
        if (epoch + 1) % args.validate_every != 0:
            continue

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        torch.set_grad_enabled(False)
        for i, (features, gens, labels) in enumerate(val_loader):
            cqts, gens, labels = features.to(device), gens.to(device), labels.to(device)

            if hasattr(model, 'use_generative_features') and model.use_generative_features:
                outputs = model(cqts, gens)
            else:
                outputs = model(cqts)

            # Flatten the outputs and labels across the batch and frames
            outputs = outputs.view(-1, outputs.shape[-1])  # (B*frames, num_classes)
            labels = labels.view(-1)  # (B*frames)

            # Compute the loss and accuracy
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        accuracy = correct / total

        print(
            f"\nEpoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}"
        )

        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

        # Save the model
        if val_loss <= min(val_losses):
            os.makedirs(args.save_dir, exist_ok=True)
            save_file = os.path.join(args.save_dir, args.save_filename)
            torch.save(model.state_dict(), save_file)

        # Reduce learning rate if not improved in the last # epochs
        if lr_scheduler is not None and args.lr_scheduler == "plateau":
            lr_scheduler.step(val_loss)

        learning_rates.append(optimiser.param_groups[0]["lr"])

        get_and_save_history(
            train_losses,
            val_losses,
            val_accuracies,
            learning_rates,
            args.save_dir,
        )

        # Early stopping if not improved in the last # epochs
        if args.early_stopping is None:
            continue

        if early_stopper(val_loss):
            print("Early stopping triggered at epoch ", epoch)
            break

    history = get_and_save_history(
        train_losses,
        val_losses,
        val_accuracies,
        learning_rates,
        args.save_dir,
    )

    return history

def get_and_save_history(
    train_losses: list,
    val_losses: list,
    val_accuracies: list,
    learning_rates: list,
    save_dir: str,
    save_filename: str = 'training_history.json'
):
    """
    Save the training history to a file.

    Args:
        history (dict): The training history.
        save_dir (str): The directory to save the history.
        save_filename (str): The filename to save the history.
    """
    os.makedirs(save_dir, exist_ok=True)
    training_history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "learning_rates": learning_rates,
    }
    write_json(training_history, f"{save_dir}/training_history.json")


def main():
    torch.manual_seed(0)

    # Load the dataset
    dataset = FixedLengthRandomChordDataset(segment_length=10, cqt_pitch_shift=True)

    # Split the dataset into train, validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    # Initialize the model
    model = CRNN(input_features=N_BINS, num_classes=NUM_CHORDS, cr2=True)

    training_args = TrainingArgs(
        epochs=100,
        lr=0.001,
        decrease_lr_epochs=10,
        decrease_lr_factor=0.1,
        early_stopping=20,
        save_model=True,
        save_dir="data/models/",
        save_filename=str(model) + "_best.pth",
        device=None,
    )

    # Train the model
    train_model(
        model,
        train_dataset,
        val_dataset,
        args=training_args,
    )


if __name__ == "__main__":
    main()
