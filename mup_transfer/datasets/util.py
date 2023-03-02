from torch.utils.data import Dataset


def get_input_shape(dataset: Dataset) -> tuple[int, ...]:
    """
    Get the input shape of a dataset. This is useful for constructing the model.

    Assumes a supervised dataset (x, y)
    """
    return dataset[0][0].shape


def get_output_size(dataset: Dataset) -> int:
    """
    Get the output size of a dataset. This is useful for constructing the model.

    Assumes a supervised dataset (x, y)
    """
    return dataset[0][1].shape[0]
