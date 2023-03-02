from enum import Enum
from functools import partial
from typing import Optional
import torch.nn as nn


class ActivationType(Enum):
    RELU = "relu"
    TANH = "tanh"
    SILU = "silu"
    GELU = "gelu"


def get_activation(activation_type: ActivationType, **activation_kwargs):
    if activation_type == ActivationType.RELU:
        return partial(nn.ReLU, **activation_kwargs)
    elif activation_type == ActivationType.TANH:
        return partial(nn.Tanh, **activation_kwargs)
    elif activation_type == ActivationType.SILU:
        return partial(nn.SiLU, **activation_kwargs)
    elif activation_type == ActivationType.GELU:
        return partial(nn.GELU, **activation_kwargs)
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")
