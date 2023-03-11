from typing import Callable
from torch import nn


def mlp_constructor(
    input_size: int,
    hidden_sizes: list[int],
    output_size: int,
    activation_constructor: Callable[[], nn.Module] = nn.ReLU,
    flattent_input: bool = True,
    bias: bool = True,
    paper_init: bool = False,  # Same init as in the original paper.
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_size, hidden_sizes[0], bias=bias)]
    for in_size, out_size in zip(hidden_sizes, hidden_sizes[1:] + [output_size]):
        layers.append(activation_constructor())
        layers.append(nn.Linear(in_size, out_size, bias=bias))
    if flattent_input:
        layers.insert(0, nn.Flatten())
    model = nn.Sequential(*layers)
    if paper_init:
        # Same init. for Standard Param. as the Tensor Programs V paper.
        model.apply(init_weights)
        nn.init.zeros_(model[-1].weight)
        if hasattr(model[-1], 'bias'):
            if model[-1].bias is not None:
                nn.init.zeros_(model[-1].bias)
    return model


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=1, mode='fan_in')

