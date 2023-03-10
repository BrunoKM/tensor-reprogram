import hydra
import numpy as np
import omegaconf
import torch
import wandb

from functools import reduce
from pathlib import Path
from mup_transfer.architectures.mlp import mlp_constructor
from hydra.core.config_store import ConfigStore

from mup_transfer.datasets.cifar10 import cifar10_constructor
from mup_transfer.config import ConfigBase
from mup_transfer.datasets.util import get_input_shape, get_output_size
from mup_transfer.loggers.wandb_logger import WandbLogger
from mup_transfer.mup.inf_types import get_inf_types, infer_inf_type_sequential_model
from mup_transfer.mup.init import mup_initialise
from mup_transfer.mup.optim_params import get_mup_sgd_param_groups
from mup_transfer.mup.utils import get_param_name


# Register the defaults from the structured dataclass config schema:
cs = ConfigStore.instance()
cs.store(name="conf", node=ConfigBase)


@hydra.main(config_path="configs/", config_name="conf", version_base=None)
def main(cfg: ConfigBase):
    """
    cfg is typed as ConfigBase for duck-typing, but during runtime it's actually an OmegaConf object.
    """
    # --- Runtime setup (logging directories, etc.)
    wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # --- Set up logging
    logger = WandbLogger(
        project="tensor-reprogram",
        entity="tensor-programs-v-reproduction",  # Log to the team's entity project
        # This is needed to make WandB and Hydra play nicely:
        settings=wandb.Settings(start_method="thread"),
    )

    # --- Construct and get the dataset
    # TODO: Make general and dependent on the config
    train_dataset, eval_datasets = cifar10_constructor(
        Path(__file__).parent.parent / "data",  # Default data directory at the root of repostiory
    )

    # --- Construct the model

    # TODO: Make general and dependent on the config
    model = mlp_constructor(
        input_size=reduce(lambda x, y: x * y, get_input_shape(train_dataset)),
        hidden_sizes=[128, 128],
        output_size=get_output_size(train_dataset),
    )

    # Initialise the model with mup
    param_inf_types = get_inf_types(
        model=model,
        input_weights_names=[get_param_name(model, model[0].weight)],  # type: ignore
        output_weights_names=[get_param_name(model, model[-1].weight)],  # type: ignore
    )
    mup_initialise(
        named_params=(named_params := list(model.named_parameters())),
        param_inf_types=param_inf_types,
        init_scale=1.0,  # TODO make configurable via config
    )
    param_groups = get_mup_sgd_param_groups(
        named_params=named_params,
        init_lr_scale=1e-3,  # TODO make configurable via config
        param_inf_types=param_inf_types,
    )


    # --- Construct the optimizer
    # TODO make optim type configurable via config.
    optim = torch.optim.SGD(
        params=param_groups,  # type: ignore
        lr=1e-3,  # TODO make configurable via config (although should be specified in all param_groups already)
    )

    # --- Compile the model
    # model_forward = torch.compile(model)

    # --- Training loop
    for epoch in range(cfg.num_epochs):
        for batch in range(10):
            logger.log_scalar("train.loss", np.random.randn())
            logger.log_scalar("train.accuracy", np.random.randn())
            logger.increment_step()

        for eval_dataset_name, eval_dataset in eval_datasets.items():
            logger.log_scalar(f"{eval_dataset_name}.loss", np.random.randn())

    # --- Save the final model

    # --- Final evaluations


if __name__ == "__main__":
    main()
