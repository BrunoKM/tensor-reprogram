from os import PathLike
from typing import Optional, Sequence
from pathlib import Path

import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from fastai.text.all import LMDataLoader


def wikitext_constructor(
    root: PathLike,
    train_batch_size: int,
    test_batch_size: int,
    bptt: int,
    shuffle: bool = False,
) -> tuple[LMDataLoader, dict[str, LMDataLoader]]:
    train_iter = WikiText2(root=str(root), split="train")
    tokenizer = get_tokenizer(None)
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>", "<eos>"])
    vocab.set_default_index(vocab["<unk>"])

    def data_process(raw_text_iter: torch.utils.data.IterableDataset) -> torch.LongTensor:
        """Converts raw text into a flat Tensor."""
        data = [
            torch.tensor(vocab(tokenizer(item) + ["<eos>"]), dtype=torch.long)
            for item in raw_text_iter
        ]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_iter, val_iter, test_iter = WikiText2(root=str(root))
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    train_dataloader = LMDataLoader([train_data], bs=train_batch_size, seq_len=bptt, shuffle=shuffle)
    valid_dataloader = LMDataLoader([val_data], bs=test_batch_size, seq_len=bptt)
    test_dataloader = LMDataLoader([test_data], bs=test_batch_size, seq_len=bptt)
    return train_dataloader, {"valid": valid_dataloader, "test": test_dataloader}


if __name__ == "__main__":
    train, eval_dict = wikitext_constructor(
        root=Path(__file__).parent.parent / "data", train_batch_size=2, test_batch_size=3, bptt=10
    )
    for ii, data in enumerate(eval_dict["test"]):
        if ii == 3:
            break
        print(data[0])
        print(data[1])
