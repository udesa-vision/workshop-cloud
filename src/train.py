"""Training module for the Fashion-MNIST CNN model.

Usage from the command line (run from the project root):
    uv run python -m src.train                          # defaults
    uv run python -m src.train --epochs 20 --data-dir data --download
    uv run python -m src.train --lr 1e-3 --weight-decay 5e-4
    uv run python -m src.train --debug                  # limit batches per epoch for quick smoke test

Programmatic usage:
    from src.data import DataConfig, get_dataloaders
    from src.models import FashionCNN
    from src.train import TrainConfig, train

    config = TrainConfig(device="cuda", num_epochs=20, checkpoint_dir="checkpoints")
    data_config = DataConfig(data_dir="data", download=True)
    model = FashionCNN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader, valid_loader, _ = get_dataloaders(data_config)

    history = train(config, model, criterion, optimizer, (train_loader, valid_loader))
    # history == {"train_loss": [...], "train_acc": [...], "valid_loss": [...], "valid_acc": [...]}
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

from src.data import DataConfig, get_dataloaders
from src.models import FashionCNN

# When train_config.debug is True, only run this many batches per epoch
# (hardcoded on purpose to keep the config small).
MAX_BATCHES_PER_EPOCH = 10


@dataclass
class TrainConfig:
    """Training hyperparameters and settings."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    checkpoint_dir: str = ".checkpoints"
    resume_path: Optional[str] = None
    debug: bool = False


@dataclass
class _CheckpointState:
    """Internal bookkeeping for checkpoints/resume."""

    start_epoch: int = 0
    best_loss: float = float("inf")


def train_one_epoch(device, loader, model, criterion, optimizer, debug: bool = False) -> Tuple[float, float]:
    """Run one training epoch. Returns (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(tqdm.tqdm(loader)):
        if debug and batch_idx >= MAX_BATCHES_PER_EPOCH:
            break

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = running_correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def valid_one_epoch(device, loader, model, criterion, debug: bool = False) -> Tuple[float, float]:
    """Run one validation epoch. Returns (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm.tqdm(loader)):
            if debug and batch_idx >= MAX_BATCHES_PER_EPOCH:
                break

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = running_correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def load_checkpoint(resume: Optional[str], device, model, optimizer=None) -> _CheckpointState:
    """Load a checkpoint if it exists. Returns (start_epoch, best_loss)."""
    state = _CheckpointState()

    if resume and os.path.exists(resume):
        print(f"loading checkpoint from: {resume}")
        checkpoint = torch.load(resume, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        state.start_epoch = checkpoint.get("epoch", 0)
        state.best_loss = checkpoint.get("valid_loss", float("inf"))

    return state


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer,
    train_loss: float,
    train_acc: float,
    valid_loss: float,
    valid_acc: float,
    is_best: bool,
) -> None:
    """Save a checkpoint to checkpoint_dir (last every epoch, best on improvement)."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, "checkpoint_last.pt"))
    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, "checkpoint_best.pt"))


def train(
    train_config: TrainConfig,
    model: nn.Module,
    criterion,
    optimizer,
    loaders: Tuple[DataLoader, DataLoader],
) -> dict:
    """Train the model and return a history dict for plotting.

    Args:
        train_config: training configuration (device, num_epochs, ...).
        model: the model to train.
        criterion: loss function.
        optimizer: optimizer.
        loaders: (train_loader, valid_loader) tuple.

    Returns:
        history dict with keys:
            "train_loss", "train_acc", "valid_loss", "valid_acc"
        each a list with one value per epoch.
    """
    train_loader, valid_loader = loaders
    device = train_config.device

    state = load_checkpoint(
        train_config.resume_path, device, model, optimizer=optimizer
    )
    start_epoch = state.start_epoch
    best_loss = state.best_loss

    history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
    }

    print("\n--- starting training ---")
    for epoch in range(start_epoch, train_config.num_epochs):
        train_loss, train_acc = train_one_epoch(
            device, train_loader, model, criterion, optimizer, debug=train_config.debug
        )
        valid_loss, valid_acc = valid_one_epoch(
            device, valid_loader, model, criterion, debug=train_config.debug
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        print(
            f"epoch [{epoch + 1}/{train_config.num_epochs}] "
            f"- train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} "
            f"- valid_loss: {valid_loss:.4f} - valid_acc: {valid_acc:.4f}"
        )

        is_best = valid_loss < best_loss
        if is_best:
            best_loss = valid_loss

        save_checkpoint(
            train_config.checkpoint_dir,
            epoch,
            model,
            optimizer,
            train_loss,
            train_acc,
            valid_loss,
            valid_acc,
            is_best=is_best,
        )

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Fashion-MNIST CNN.")
    parser.add_argument("--epochs", type=int, default=32, help="number of epochs (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="L2 regularization weight decay (default: 5e-4)")
    parser.add_argument("--data-dir", type=str, default=".data", help="directory with FashionMNIST data, relative to the project root (default: .data)")
    parser.add_argument("--download", action="store_true", help="download the FashionMNIST dataset if not present")
    parser.add_argument("--batch-size", type=int, default=64, help="dataloader batch size (default: 64)")
    parser.add_argument("--checkpoint-dir", type=str, default=".checkpoints", help="where to save checkpoints (default: .checkpoints)")
    parser.add_argument("--resume", type=str, default=None, help="path to a checkpoint to resume training from")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help="device to train on (default: cuda if available else cpu)")
    parser.add_argument("--debug", action="store_true", help="only run a few batches per epoch for a quick smoke test")
    args = parser.parse_args()

    config = TrainConfig(
        device=args.device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        resume_path=args.resume,
        debug=args.debug,
    )

    data_config = DataConfig(
        data_dir=args.data_dir,
        download=args.download,
        batch_size=args.batch_size,
    )

    model = FashionCNN().to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_loader, valid_loader, _ = get_dataloaders(data_config)

    history = train(config, model, criterion, optimizer, (train_loader, valid_loader))

    # Brief summary so results are visible without needing to inspect the history object.
    print("\n--- training finished ---")
    print(f"final train_loss: {history['train_loss'][-1]:.4f} - train_acc: {history['train_acc'][-1]:.4f}")
    print(f"final valid_loss: {history['valid_loss'][-1]:.4f} - valid_acc: {history['valid_acc'][-1]:.4f}")
    print(f"best valid_loss:  {min(history['valid_loss']):.4f}")


if __name__ == "__main__":
    main()