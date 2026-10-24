"""Run reporting utilities: parameter snapshot, per-epoch CSV, and curve plots.

Every training run gets its own folder under the checkpoint directory. This
module writes three artifacts into that folder:

    run_report.txt   - parameter snapshot of the run (appended on resume)
    history.csv      - one row per completed epoch (loss, accuracy, lr)
    curves.png       - loss/accuracy curves, regenerated after every epoch

Plotting uses the "Agg" backend so it works in headless environments.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for headless runs
import matplotlib.pyplot as plt  # noqa: E402

from src.data import DataConfig  # noqa: E402


def write_run_report(
    run_dir: str,
    config,
    data_config: Optional[DataConfig] = None,
    model=None,
    optimizer=None,
    criterion=None,
) -> str:
    """Write (or append to) run_report.txt with the parameters of this run.

    A fresh run writes the report from scratch; a resumed run appends a
    timestamped block instead of overwriting, preserving the full history.

    Returns the path of the report file.
    """
    os.makedirs(run_dir, exist_ok=True)
    report_path = os.path.join(run_dir, "run_report.txt")

    lines = []
    if os.path.exists(report_path):
        lines.append("")
        lines.append("=" * 50)
        lines.append(f"--- resumed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        lines.append("=" * 50)
        lines.append("")
    else:
        lines.append(f"run created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("-" * 50)
        lines.append("")

    lines.append("TrainConfig:")
    lines.append(f"  device:            {config.device}")
    lines.append(f"  num_epochs:        {config.num_epochs}")
    lines.append(f"  learning_rate:     {config.learning_rate}")
    lines.append(f"  weight_decay:      {config.weight_decay}")
    lines.append(f"  patience:          {config.patience}")
    lines.append(f"  checkpoint_dir:    {config.checkpoint_dir}")
    lines.append(f"  resume_path:       {config.resume_path}")
    lines.append(f"  debug:             {config.debug}")
    lines.append(f"  seed:              {config.seed}")

    if data_config is not None:
        lines.append("")
        lines.append("DataConfig:")
        lines.append(f"  data_dir:          {data_config.data_dir}")
        lines.append(f"  download:          {data_config.download}")
        lines.append(f"  batch_size:        {data_config.batch_size}")
        lines.append(f"  valid_ratio:       {data_config.valid_ratio}")

    if model is not None:
        lines.append("")
        lines.append(f"model:               {model.__class__.__name__}")
    if optimizer is not None:
        lines.append(f"optimizer:           {optimizer.__class__.__name__}")
    if criterion is not None:
        lines.append(f"criterion:           {criterion.__class__.__name__}")

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return report_path


def append_history_csv(
    run_dir: str,
    epoch: int,
    train_loss: float,
    train_acc: float,
    valid_loss: float,
    valid_acc: float,
    lr: float,
) -> str:
    """Append one epoch row to history.csv in run_dir.

    The header is written only when the file is first created, so resumed
    runs append cleanly. Returns the path of the CSV file.
    """
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "history.csv")

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc", "lr"])
        writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{valid_loss:.6f}", f"{valid_acc:.6f}", f"{lr:.2e}"])

    return csv_path


def load_history_csv(run_dir: str) -> Optional[dict]:
    """Load history.csv from run_dir into a history dict, or None if absent.

    Used when resuming so the CSV and plots reflect the full run across the
    original + resumed epochs.
    """
    csv_path = os.path.join(run_dir, "history.csv")
    if not os.path.exists(csv_path):
        return None

    history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
    }
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            history["train_loss"].append(float(row["train_loss"]))
            history["train_acc"].append(float(row["train_acc"]))
            history["valid_loss"].append(float(row["valid_loss"]))
            history["valid_acc"].append(float(row["valid_acc"]))
    return history


def plot_history(run_dir: str, history: dict) -> str:
    """Plot train/valid loss and accuracy curves into curves.png in run_dir.

    The figure is overwritten each call, so it always reflects the latest
    state of ``history``. Returns the path of the saved PNG.
    """
    os.makedirs(run_dir, exist_ok=True)
    plot_path = os.path.join(run_dir, "curves.png")

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train_loss")
    axes[0].plot(epochs, history["valid_loss"], marker="o", label="valid_loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Curves")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Accuracy curves
    axes[1].plot(epochs, history["train_acc"], marker="o", label="train_acc")
    axes[1].plot(epochs, history["valid_acc"], marker="o", label="valid_acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    return plot_path