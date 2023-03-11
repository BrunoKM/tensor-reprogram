from dataclasses import dataclass, field
import enum
from typing import Any


class ArchitectureType(str, enum.Enum):
    MLP = "mlp"
    TRANSFORMER = "transformer"


@dataclass
class ArchitectureConfig:
    bias: bool = True


@dataclass
class DatasetConfig:
    name: DatasetType


@dataclass
class DataLoaderConfig:
    train_batch_size: int = 128
    eval_batch_size: int = 512
    num_workers: int = 4
    pin_memory: bool = False


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
class ConfigBase:

    architecture: ArchitectureConfig
    dataset: DatasetConfig
    optimisation: OptimizerConfig
    initialisation: InitialisationConfig
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)


@dataclass
class ArchitectureTransformerConfig(ArchitectureConfig):
    bias: bool = False
    d_model: int = 256
    ffn_ratio: int = 1
    nlayers: int = 2
    nhead: int = 2
    dropout: float = 0.2
    tied: bool = False
    init_var: float = 1
    ntokens: int = 33278
    attn_mult: int = 1
    output_mul: int = 1
    standparam: bool = False    # TODO


@dataclass
class ConfigTransformer(ConfigBase):
    num_epochs: int = 40
    bptt: int = 35

    architecture: ArchitectureConfig = field(default_factory=ArchitectureTransformerConfig)
