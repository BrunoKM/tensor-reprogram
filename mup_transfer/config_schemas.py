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
    lr: float = 1e-3
    # If specified, overrides the "global" lr with a per-parameter learning rate.
    per_param_lr: dict[str, float] = field(default_factory=dict)
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    clip_grad: float = float("inf")


@dataclass
class InitialisationConfig:
    init_scale: float = 1.0
    # If specified, overrides the "global" init_scale with a per-parameter init_scale
    init_scales_per_param: dict[str, float] = field(default_factory=dict)


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
    att_mult: float = 1.
    attn_mult: float = 1.
    output_mul: float = 1.


@dataclass
class MLPArchitectureConfig:
    add_bias: bool = True
    hidden_sizes: Optional[list[int]] = None
    paper_init: bool = False
    width: Optional[int] = None
    depth: Optional[int] = None


@dataclass
class ConfigBase:
    num_epochs: int
    architecture_type: ArchitectureType
    dataset_type: DatasetType
    use_mu_param: bool = True

    optimization: OptimizerConfig = field(default_factory=OptimizerConfig)
    initialisation: InitialisationConfig = field(default_factory=InitialisationConfig)
    transformer_config: TransformerArchitectureConfig = field(default_factory=TransformerArchitectureConfig)
    mlp_config: MLPArchitectureConfig = field(default_factory=MLPArchitectureConfig)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)

    log_to_wandb: bool = True
    wandb_project_name: str = "tensor-reprogram"
