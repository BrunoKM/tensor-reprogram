from collections import OrderedDict
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
    layers: list[tuple[str, nn.Module]] = [("input_layer", nn.Linear(input_size, hidden_sizes[0], bias=bias))]
    for i in range(len(hidden_sizes) - 1):
        in_size, out_size = hidden_sizes[i : i + 2]
        layers.append((f"activation{i}", activation_constructor()))
        layers.append((f"hidden_layer{i}", nn.Linear(in_size, out_size, bias=bias)))
    layers.append((f"activation{len(hidden_sizes)}", activation_constructor()))
    layers.append((f"output_layer", nn.Linear(hidden_sizes[-1], output_size, bias=bias)))
    if flattent_input:
        layers.insert(0, ("input_flatten", nn.Flatten()))

    layers_dict = OrderedDict(layers)
    model = nn.Sequential(layers_dict)

    if paper_init:
        # Same init. for Standard Param. as the Tensor Programs V paper.
        model.apply(init_weights)
        nn.init.zeros_(layers_dict["output_layer"].weight)
        if hasattr(layers_dict["output_layer"], "bias"):
            if layers_dict["output_layer"].bias is not None:
                nn.init.zeros_(layers_dict["output_layer"].bias)
    return model


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=1, mode="fan_in")
