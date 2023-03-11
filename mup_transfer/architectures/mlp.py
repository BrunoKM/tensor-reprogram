from typing import Callable
from torch import nn


def mlp_constructor(
    input_size: int,
    hidden_sizes: list[int],
    output_size: int,
    activation_constructor: Callable[[], nn.Module] = nn.ReLU,
    flattent_input: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_size, hidden_sizes[0])]
    for in_size, out_size in zip(hidden_sizes, hidden_sizes[1:] + [output_size]):
        layers.append(activation_constructor())
        layers.append(nn.Linear(in_size, out_size))
    if flattent_input:
        layers.insert(0, nn.Flatten())
    return nn.Sequential(*layers)
