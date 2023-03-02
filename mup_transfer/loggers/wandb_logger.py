from typing import Any, Optional
from mup_transfer.loggers.logger import LoggerBase


class WandbLogger(LoggerBase):
    def __init__(self, **wandb_params) -> None:
        super().__init__()
        import wandb
        wandb.init(**wandb_params)
        self._wandb = wandb

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        self._wandb.log({name: value}, step=step or self._step)

    def __del__(self):
        self._wandb.finish()
