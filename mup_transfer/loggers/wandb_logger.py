from typing import Any, Optional

import wandb
from mup_transfer.loggers.logger import LoggerBase


class WandbLogger(LoggerBase):
    def __init__(self, **wandb_params) -> None:
        super().__init__()
        import wandb
        self._run = wandb.init(**wandb_params)

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        wandb.log({name: value}, step=step or self._step)

    def __del__(self):
        wandb.finish()
