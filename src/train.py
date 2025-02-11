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

from src.models.ismir2017 import ISMIR2017ACR
from src.data.dataset import FixedLengthRandomChordDataset, FixedLengthChordDataset
from src.utils import get_torch_device, collate_fn, NUM_CHORDS, EarlyStopper


class TrainingArgs:
    def __init__(
        self,
        num_epochs: int,
        lr: float,
        batch_size: int = 64,
        segment_length: int = 10,
        early_stopping: int = 20,
        decrease_lr_factor: float = 0.1,
        decrease_lr_epochs: int = 10,
        train_on_X: bool = True,
        use_weighted_loss: bool = False,
        do_validation: bool = True,
        save_model: bool = True,
        save_dir: str = "data/models/",
        save_filename: str = "best_model.pth",
        device: torch.device = None,
    ):
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.segment_length = segment_length
        self.early_stopping = early_stopping
        self.decrease_lr_factor = decrease_lr_factor
        self.decrease_lr_epochs = decrease_lr_epochs
        self.train_on_X = train_on_X
        self.use_weighted_loss = use_weighted_loss
        self.do_validation = do_validation
        self.save_model = save_model
        self.save_dir = save_dir
        self.save_filename = save_filename
        self.device = device

    def _asdict(self):
        return {
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "segment_length": self.segment_length,
            "early_stopping": self.early_stopping,
            "decrease_lr_factor": self.decrease_lr_factor,
            "decrease_lr_epochs": self.decrease_lr_epochs,
            "do_validation": self.do_validation,
            "train_on_X": self.train_on_X,
            "use_weighted_loss": self.use_weighted_loss,
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    if args.early_stopping is not None:
        early_stopper = EarlyStopper(args.early_stopping)

    if args.device is None:
        # Use GPU if available, check for cuda and mps
        device = get_torch_device()
        print("Using device:", device)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.decrease_lr_factor,
        patience=args.decrease_lr_epochs,
    )
    prev_lr = args.lr

    val_losses = []
    val_accuracies = []
    train_losses = []

    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        train_loss = 0.0
        for i, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)

            # Flatten the outputs and labels
            outputs = outputs.view(-1, outputs.shape[-1])  # (B*frames, num_classes)
            labels = labels.view(-1)  # (B*frames)

            # Compute the loss and backpropagate
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        if not args.do_validation:
            print(f"\nEpoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}")
            continue

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (features, labels) in tqdm(enumerate(val_loader)):
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)

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
                f"\nEpoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}"
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(accuracy)

            # Save the model
            if args.save_model:

                def save_model():
                    os.makedirs(args.save_dir, exist_ok=True)
                    save_file = os.path.join(args.save_dir, args.save_filename)
                    torch.save(model.state_dict(), save_file)

                if not args.do_validation and train_loss <= min(
                    train_losses
                ):  # Save the best model based on training loss
                    save_model()
                elif args.do_validation and val_loss <= min(
                    val_losses
                ):  # Save the best model based on validation loss
                    save_model()

            # Reduce learning rate if not improved in the last # epochs
            if args.decrease_lr_epochs is not None:
                scheduler.step(val_loss)
                if optimizer.param_groups[0]["lr"] < prev_lr:
                    prev_lr = optimizer.param_groups[0]["lr"]
                    print(f"Reducing learning rate to {prev_lr}")

            # Early stopping if not improved in the last # epochs
            if args.early_stopping is None:
                continue
            elif early_stopper(val_loss):
                print("Early stopping triggered at epoch ", epoch)
                break

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    }
    return history


def main():
    torch.manual_seed(0)

    # Load the dataset
    dataset = FixedLengthRandomChordDataset(
        segment_length=10, hop_length=4096, cached=True, random_pitch_shift=True
    )

    # Split the dataset into train, validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    # Initialize the model
    model = ISMIR2017ACR(
        input_features=dataset.full_dataset.n_bins, num_classes=NUM_CHORDS, cr2=True
    )

    training_args = TrainingArgs(
        num_epochs=100,
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
