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
    batch_size: int,
    bptt: int,
) -> tuple[Dataset, dict[str, Dataset]]:
    train_iter = WikiText2(root, split='train')
    tokenizer = get_tokenizer(None)
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>', '<eos>'])
    vocab.set_default_index(vocab['<unk>'])
    # print(len(vocab))

    def data_process(raw_text_iter: torch.utils.data.IterableDataset) -> torch.LongTensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(vocab(tokenizer(item) + ['<eos>']), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    train_dataloader = LMDataLoader([train_data], bs=batch_size, seq_len=bptt)
    valid_dataloader = LMDataLoader([val_data], bs=batch_size, seq_len=bptt)
    test_dataloader = LMDataLoader([test_data], bs=batch_size, seq_len=bptt)
    return train_dataloader, {'valid': valid_dataloader, "test": test_dataloader}


if __name__=='__main__':
    train, eval_dict = wikitext_constructor(Path(__file__).parent.parent / "data", 2, 10)
    for ii, data in enumerate(eval_dict['test']):
        if ii == 3:
            break
        print(data[0])
        print(data[1])
