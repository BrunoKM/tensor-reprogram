from dataclasses import dataclass, field
import enum
from typing import Any


@dataclass
class ArchitectureConfig:
    pass


@dataclass
class DatasetConfig:
    pass


class OptimizerType(str, enum.Enum):
    SGD = "SGD"
    ADAM = "ADAM"


@dataclass
class OptimizerConfig:
    optimizer_type: OptimizerType = OptimizerType.SGD
    lr: float = 1e-3
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigBase:
    num_epochs: int = 10

    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimisation: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class ArchitectureTransformerConfig:
    bias: bool = False
    d_model: int = 256
    ffn_ratio: int = 1
    nlayers: int = 2
    nhead: int = 2
    dropout: float = 0.2
    tied: bool = False
    init_var: float = 1
    ntokens: int = xx
    att_mult: int = 1
    output_mul: int = 1
    standparam: bool = False    # TODO


@dataclass
class ConfigTransformer:
    num_epochs: int = 40
    batch_size: int = 20
    bptt: int = 35
    seed: int = 1111

    architecture: ArchitectureConfig = field(default_factory=ArchitectureTransformerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimisation: OptimizerConfig = field(default_factory=OptimizerConfig)
