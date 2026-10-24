"""Data loading utilities for the Fashion-MNIST dataset."""

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.paths import PROJECT_ROOT


@dataclass
class DataConfig:
    """Configuration for data loading.

    `data_dir` is interpreted relative to the project root.
    """

    data_dir: str = ".data"
    download: bool = False
    batch_size: int = 64
    valid_ratio: float = 0.8

    @property
    def dataset_root(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p


def get_dataloaders(config: DataConfig):
    # Augmentation is applied to training data only; the validation/test
    # split shares the same deterministic preprocessing.
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        root=config.dataset_root, train=True, download=config.download, transform=train_transform
    )
    # Separate instance so validation is NOT augmented (deterministic eval).
    # torchvision reuses the already-downloaded files, so this costs no re-download.
    valid_dataset = datasets.FashionMNIST(
        root=config.dataset_root, train=True, download=config.download, transform=eval_transform
    )
    test_dataset = datasets.FashionMNIST(
        root=config.dataset_root, train=False, download=config.download, transform=eval_transform
    )

    train_size = int(config.valid_ratio * len(train_dataset))
    train_set = Subset(train_dataset, range(0, train_size))
    valid_set = Subset(valid_dataset, range(train_size, len(valid_dataset)))

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader