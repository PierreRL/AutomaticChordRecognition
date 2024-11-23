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
from src.data.dataset import FixedLengthChordDataset
from src.utils import get_torch_device


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    early_stopping: int = 10,
    do_validation: bool = True,
    save_model: bool = True,
    save_dir: str = "data/models/",
    device: torch.device = None,
):
    """
    Train a model on the given data.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        num_epochs (int): Number of epochs to train for.
        lr (float): Learning rate for the optimizer.
    """

    if not device:
        # Use GPU if available, check for cuda and mps
        device = get_torch_device()
        print("Using device:", device)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    val_losses = []

    for epoch in tqdm(range(num_epochs)):
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

        if not do_validation:
            print(f"\nEpoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
            continue

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (features, labels) in enumerate(val_loader):
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
                f"\nEpoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}"
            )

            val_losses.append(val_loss)

            # Save the best model
            if not val_losses or val_loss < min(val_losses):
                print("Saving model")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

            # Early stopping if not improved in the last 10 epochs
            if not early_stopping:
                continue
            if len(val_losses) > early_stopping and val_loss > max(
                val_losses[-early_stopping:]
            ):
                print("Early stopping triggered at epoch ", epoch)
                break


def main():
    torch.manual_seed(0)

    # Load the dataset
    dataset = FixedLengthChordDataset(frame_length=10, cached=True)

    # Split the dataset into train, validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize the model
    model = ISMIR2017ACR(input_features=dataset.n_bins, num_classes=25)

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=1, lr=0.001)


if __name__ == "__main__":
    main()
