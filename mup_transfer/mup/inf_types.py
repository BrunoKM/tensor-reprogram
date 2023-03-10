"""
The different "types" of parameters of the model. These will be initialised differently.,
and will have their optimiser's parameters scaled differently, depending on how they behave
in the infinite width limit.
"""

from enum import Enum
from operator import itemgetter
from typing import Sequence, Union

import torch
import torch.nn as nn


class InfType(Enum):
    """
    See "Tensor Programs V: Tuning Large Neural Networks via Zero-shot Hyperparameter Transfer"
    - Table 3 by Greg Yang for details.
    """
    INPUT_OR_BIAS = "input_or_bias"
    HIDDEN_WEIGHT = "hidden_weight"
    OUTPUT_WEIGHT = "output_weight"


def infer_inf_type_sequential_model(model: torch.nn.Sequential) -> dict[str, InfType]:
    """
    Given a model, return a dictionary mapping the names of the parameters to their
    infinite width type.

    The function makes some assumptions:
    - The first parameter with at least 2 dimensions is the input layer weight
    - The last parameter with at least 2 dimensions is the output layer weight
    """
    # This very dodgily assumes that param. names for a sequential model start with the index of the module, and hence
    # sorting them by name will give the correct order of the modules.
    named_params: list[tuple[str, nn.Parameter]] = sorted(
        list(model.named_parameters()),
        key=itemgetter(0),  # Sort by param. name
    )
    weight_names = [name for name, param in named_params if len(param.shape) >= 2]

    input_layer_weight_name = weight_names[0]
    output_layer_weight_name = weight_names[-1]
    return get_inf_types(model, [input_layer_weight_name], [output_layer_weight_name])


def get_inf_types(model: torch.nn.Module, input_weights_names: Sequence[str], output_weights_names: Sequence[str]) -> dict[str, InfType]:
    """
    Given a model, and a manual specification by the user of which parameters are input weights and
    which are output weights, return a dictionary mapping the names of the parameters to their
    infinite width type.
    """
    named_params: list[tuple[str, nn.Parameter]] = list(model.named_parameters())
    bias_names = [name for name, param in named_params if len(param.shape) <= 1]

    # Get a mapping from inf. type to list of param. names of that inf. type
    inf_type_groups: dict[InfType, list[str]] = {
        InfType.INPUT_OR_BIAS: list(input_weights_names) + bias_names,
        InfType.OUTPUT_WEIGHT: list(output_weights_names),
        InfType.HIDDEN_WEIGHT: [name for name, param in named_params if name not in {*set(input_weights_names), *set(output_weights_names), *bias_names}],
    }
    # Invert the mapping to be a mapping from param. name to inf. type
    return {name: inf_type for inf_type, names in inf_type_groups.items() for name in names}

