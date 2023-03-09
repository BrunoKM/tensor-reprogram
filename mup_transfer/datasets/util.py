from typing import Union
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST


def get_input_shape(dataset: Dataset) -> tuple[int, ...]:
    """
    Get the input shape of a dataset. This is useful for constructing the model.

    Assumes a supervised dataset (x, y)
    """
    return dataset[0][0].shape


def get_output_size(dataset: Union[CIFAR10, MNIST]) -> int:
    """
    Get the output size of a dataset. This is useful for constructing the model.

    Assumes a supervised dataset (x, y)
    """
    # TODO: Generalise for text data.
    return len(dataset.classes)
