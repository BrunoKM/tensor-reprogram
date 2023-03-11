from os import PathLike
from typing import Optional, Sequence

from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms


def cifar10_constructor(
    root: PathLike,
    subset_idxs: Optional[Sequence[int]] = None,
) -> tuple[Dataset, dict[str, Dataset]]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = CIFAR10(root=str(root), train=True, download=True, transform=transform)
    test_set = CIFAR10(root=str(root), train=False, download=True, transform=transform)
    if subset_idxs is not None:
        train_set = Subset(train_set, subset_idxs)
        test_set = Subset(test_set, subset_idxs)
    return train_set, {"test": test_set}
