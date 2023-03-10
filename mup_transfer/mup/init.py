"""
Initialisation functions for mu-parameterisation. These follow the parameterisation in Table 3 of
"Tensor Programs V: Tuning Large Neural Networks via Zero-shot Hyperparameter Transfer". This is
because using the parameterisations in Table 8 requires altering the model with additional 
scalar multipliers.
"""
from typing import Sequence, Union
import torch.nn as nn

from mup_transfer.mup.inf_types import InfType


def mup_initialise(
    named_params: Sequence[tuple[str, nn.Parameter]],
    param_inf_types: dict[str, InfType],
    init_scale: Union[float, dict[str, float]] = 1.0,
) -> None:
    """
    In-place initialise the parameters of a model using the MUP initialisation scheme described in
    "Tensor Programs V: Tuning Large Neural Networks via Zero-shot Hyperparameter Transfer".

    Note: The parameters are ASSUMED to have shape (fan_out, fan_in, ...) or (fan_out,)!! 
        The latter is the case for biases.

    Args:
        named_params: A sequence of (name, param) pairs, where name is the name of the parameter, and param is the parameter itself.
        param_inf_types: A dictionary mapping the names of the parameters to their infinite width type. The initialisation scheme
            will be different for each type of parameter.
        init_scale: The scale of the initialisation. This is a tunable hyperparameter constant that is independent of scale. If a dictionary,
            then the keys should be the names of the parameters, and the values should be the scale for that parameter.
    """
    # Input checks:
    if len(param_inf_types) != len(named_params):
        inf_type_nameset = set(param_inf_types.keys())
        named_params_nameset = set(name for name, _ in named_params)
        raise ValueError(
            f"The parameters in param_inf_types do not match the parameters in named_params.\n"
            f"The extra parameters in param_inf_types are: {inf_type_nameset - named_params_nameset}\n"
            f"The extra parameters in named_params are: {named_params_nameset - inf_type_nameset}"
        )

    # Initialise all params
    for name, param in named_params:
        inf_type = param_inf_types[name]
        init_scale_for_param = init_scale[name] if isinstance(init_scale, dict) else init_scale

        mup_initialise_param(param, inf_type=inf_type, init_scale=init_scale_for_param)


def mup_initialise_param(param: nn.Parameter, inf_type: InfType, init_scale: float) -> None:
    """
    In-place initialise a parameter using the MUP initialisation scheme described in
    "Tensor Programs V: Tuning Large Neural Networks via Zero-shot Hyperparameter Transfer", as described
    in Table 3.

    Args:
        param: The parameter to initialise.
        inf_type: The infinite width type of the parameter.
        init_scale: The scale of the initialisation. This is a tunable hyperparameter constant that is independent of scale.
    """
    # Fan-in of a bias is 1
    fan_in = param.shape[1] if len(param.shape) >= 2 else 1

    match inf_type:
        case InfType.INPUT_OR_BIAS | InfType.HIDDEN_WEIGHT:
            scale_multiplier = (1 / fan_in) ** 0.5
        case InfType.OUTPUT_WEIGHT:
            scale_multiplier = (1 / fan_in)
        case _:
            raise ValueError(f"Unrecognised infinite width type: {inf_type}")
    # Initialise in place.
    nn.init.normal_(param, mean=0.0, std=init_scale * scale_multiplier)
