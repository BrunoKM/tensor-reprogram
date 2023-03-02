import torch
from mup_transfer.config import OptimizerConfig, OptimizerType


def optimizer_constructor(optimizer_config: OptimizerConfig):
    if optimizer_config.optimizer_type == OptimizerType.SGD:
        return torch.optim.SGD(**optimizer_config.optimizer_kwargs)