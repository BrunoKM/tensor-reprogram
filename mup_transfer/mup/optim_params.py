"""
Compute adequate scalings for the learning rate and other optimiser parameters for each of the parameter groups
in the mu-parameterisation given in:
    "Tensor Programs V: Tuning Large Neural Networks via Zero-shotHyperparameter Transfer" by Greg Yang
"""

from typing import Sequence, TypedDict, Union
import torch
import torch.nn as nn

from mup_transfer.mup.inf_types import InfType


class ParamGroup(TypedDict):
    params: list[nn.Parameter]
    lr: float


class SGDParamGroup(ParamGroup):
    weight_decay: float


class AdamParamGroup(SGDParamGroup):
    betas: tuple[float, float]
    eps: float


def get_mup_sgd_param_groups(
        named_params: Sequence[tuple[str, nn.Parameter]],
        init_lr_scale: Union[float, dict[str, float]],
        param_inf_types: dict[str, InfType],
    ) -> list[SGDParamGroup]:
    """
    # TODO: Make it work with weight decay somehow

    Every parameter ends up being its own group, but hopefully that is not a bottleneck. If it is, a fix is to
    join all the param_groups that have the same optim. hyperparameters post-hoc.

    Args:
        named_params: 
        init_lr_scale: 
        param_inf_types: A dictionary mapping the names of the parameters to their infinite width type. The 
            initialisation scheme will be different for each type of parameter.
    """
    mup_param_groups: list[SGDParamGroup] = []
    for param_name, param in named_params:
        init_lr = init_lr_scale if isinstance(init_lr_scale, float) else init_lr_scale[param_name]
        # Fan-in of a bias is 1
        fan_out = param.shape[0]
        fan_in = param.shape[1] if len(param.shape) >= 2 else 1
        inf_type = param_inf_types[param_name]

        match inf_type:
            case InfType.INPUT_OR_BIAS:
                lr_multiplier = fan_out
            case InfType.OUTPUT_WEIGHT:
                lr_multiplier = 1 / fan_in
            case InfType.HIDDEN_WEIGHT:
                lr_multiplier = 1
            case _:
                raise ValueError(f"Unrecognised infinite width type: {inf_type}")
        lr = init_lr * lr_multiplier
        # Split the parameters in group by inf. type
        mup_param_groups.append(
            SGDParamGroup(params=[param], lr=lr, weight_decay=0.0)
        )
    return mup_param_groups
