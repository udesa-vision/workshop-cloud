"""Reproducibility utilities for deterministic training."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed all random number generators for reproducible runs.

    Seeds Python's ``random``, NumPy, and PyTorch (CPU + CUDA), disables
    cuDNN's benchmarking mode, and forces deterministic cuDNN algorithms.
    Call this once at the start of the program, before building the model,
    optimizer, or dataloaders.

    Note: on CUDA, a handful of PyTorch ops remain non-deterministic even
    with these settings; the ops used by ``FashionCNN`` are all covered.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False