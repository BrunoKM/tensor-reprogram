from dataclasses import dataclass, field
import enum
from typing import Any, Optional, Union


class ArchitectureType(str, enum.Enum):
    MLP = "MLP"
    TRANSFORMER = "TRANSFORMER"


class DatasetType(str, enum.Enum):
    CIFAR10 = "CIFAR10"
    WIKITEXT = "WIKITEXT"


@dataclass
class DataLoaderConfig:
    train_batch_size: int = 128
    eval_batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = False
    bptt: int = 35  # For sequence input


class OptimizerType(str, enum.Enum):
    SGD = "SGD"
    ADAM = "ADAM"


@dataclass
class OptimizerConfig:
    optimizer_type: OptimizerType = OptimizerType.SGD
    default_lr: float = 1e-3
    """Default learning rate if a param specific lr is not specified"""
    global_lr: float = 1.0  # Multiplier for all learning rates
    # If specified, overrides the "global" lr with a per-parameter learning rate.
    per_param_lr: dict[str, float] = field(default_factory=dict)
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    clip_grad: float = float("inf")


class DistributionType(str, enum.Enum):
    NORMAL = "NORMAL"
    UNIFORM = "UNIFORM"


class ParameterisationType(str, enum.Enum):
    """
    The parameterisation of the initialisation scales and learning rates.

    This only determines how the scales and learning rates are _scaled_ with the width of each part of the network!

    SP and PYTORCH do not apply any scaling to the learning rates, whereas MUP does.

    SP and PYTORCH differ in initialisation scaling of biases (see InitialisationConfig for more details).
    """
    MUP = "MUP"
    """mu-parameterisation"""
    SP = "SP"
    """Standard Parameterisation"""
    PYTORCH = "PYTORCH"
    """PyTorch default initialisation scaling (differs from SP in how bias initialisation is scaled)"""


@dataclass
class InitialisationConfig:
    """
    Configuration for initialisation of the model parameters.

    To recover the default PyTorch initialisation, set:
     - default_init_scale = 1 / sqrt(3) ≈ 0.577
     - init_distribution = DistributionType.UNIFORM
     - parameterisation: ParameterisationType.PYTORCH

    (Note that the biases in the PYTORCH init. are scaled with 1 / sqrt(layer_fan_in), where the
    layer_fan_in above refers to the number of inputs to the layer after which biases are added.
    In Tensor Programs V, and in this repo, the fan_in of a bias is in contrast taken to be 1)
    (the default PyTorch initialisation initialises everything as Uniform(-1/sqrt(fan_in)), 1/sqrt(fan_in), where
    fan_in is the layer fan_in - number of inputs to the layer)

    Scale throughout refers to the "base" standard deviation of the distribution (and not e.g. bounds of the uniform).
    So, for example, when using Standard Parameterisation (SP) the standard deviation of the distribution to sample from
    for a weight matrix will be `scale / sqrt(fan_in)`
    """

    default_init_scale: float = 1.0
    """Default init scale if a param specific init_scale is not specified."""

    global_init_scale: float = 1.0
    """Multiplier to apply tor all init scales (including default)"""

    init_scales_per_param: dict[str, float] = field(default_factory=dict)
    """If specified, overrides the default init_scale with a per-parameter init_scale"""

    init_distribution: DistributionType = DistributionType.NORMAL
    """The initialisation distribution"""


@dataclass
class TransformerArchitectureConfig:
    add_bias: bool = False
    d_model: int = 256
    ffn_ratio: int = 1
    nlayers: int = 2
    nhead: int = 2
    dropout: float = 0.2
    tied: bool = False
    init_var: float = 1
    ntokens: int = 33278
    att_mult: float = 1.0
    attn_mult: float = 1.0
    output_mul: float = 1.0


@dataclass
class MLPArchitectureConfig:
    add_bias: bool = True
    hidden_sizes: Optional[list[int]] = None
    width: Optional[int] = None
    depth: Optional[int] = None


@dataclass
class ConfigBase:
    num_epochs: int
    architecture_type: ArchitectureType
    dataset_type: DatasetType
    parameterisation: ParameterisationType = ParameterisationType.MUP

    optimization: OptimizerConfig = field(default_factory=OptimizerConfig)
    initialisation: InitialisationConfig = field(default_factory=InitialisationConfig)
    transformer_config: TransformerArchitectureConfig = field(
        default_factory=TransformerArchitectureConfig
    )
    mlp_config: MLPArchitectureConfig = field(default_factory=MLPArchitectureConfig)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)

    log_to_wandb: bool = True
    wandb_project_name: str = "tensor-reprogram"
