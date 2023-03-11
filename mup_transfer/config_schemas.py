from dataclasses import dataclass, field
import enum
from typing import Any


class ArchitectureType(str, enum.Enum):
    MLP = "mlp"
    TRANSFORMER = "transformer"


class DatasetType(str, enum.Enum):
    CIFAR10 = "cifar10"
    WIKITEXT = "wikitext"


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
    optimizer_type: OptimizerType
    lr: float = 1e-3
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class InitialisationConfig:
    init_scale: float = 1.0


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
    att_mult: float = 1
    output_mul: float = 1

@dataclass
class MLPArchitectureConfig:
    add_bias: bool = True
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])


@dataclass
class ConfigBase:
    num_epochs: int
    architecture_type: ArchitectureType
    dataset_type: DatasetType
    optimisation: OptimizerConfig
    initialisation: InitialisationConfig
    transformer_config: TransformerArchitectureConfig = field(default_factory=TransformerArchitectureConfig)
    mlp_config: MLPArchitectureConfig = field(default_factory=MLPArchitectureConfig)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
