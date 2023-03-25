from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def get_data_loaders(
    train_dataset: Dataset,
    eval_datasets: dict[str, Dataset],
    train_batch_size: int = 128,
    eval_batch_size: int = 512,
    num_workers: int = 4,
    pin_memory: bool = False,
    shuffle: bool = True,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    eval_loaders = dict()
    for eval_dataset_name, eval_dataset in eval_datasets.items():
        eval_loaders[eval_dataset_name] = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return train_loader, eval_loaders
