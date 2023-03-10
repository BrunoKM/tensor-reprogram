from dataclasses import dataclass, field
import enum
from typing import Any


@dataclass
class ArchitectureConfig:
    pass


@dataclass
class DatasetConfig:
    pass


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
    optimizer_type: OptimizerType = OptimizerType.SGD
    lr: float = 1e-3
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class InitialisationConfig:
    init_scale: float = 1.0


@dataclass
class ConfigBase:
    num_epochs: int = 10

    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    optimisation: OptimizerConfig = field(default_factory=OptimizerConfig)
    initialisation: InitialisationConfig = field(default_factory=InitialisationConfig)
