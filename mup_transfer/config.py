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

