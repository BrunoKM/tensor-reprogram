from os import PathLike
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


from typing import Optional, Sequence


def cifar10_constructor(
    root: PathLike,
    subset_idxs: Optional[Sequence[int]] = None,
) -> tuple[Dataset, dict[str, Dataset]]:
    train_set = CIFAR10(root=str(root), train=True, download=True, transform=ToTensor())
    test_set = CIFAR10(root=str(root), train=True, download=True, transform=ToTensor())
    if subset_idxs is not None:
        train_set = Subset(train_set, subset_idxs)
        test_set = Subset(test_set, subset_idxs)
    return train_set, {"test": test_set}
