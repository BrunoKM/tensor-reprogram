from functools import reduce
import hydra
import omegaconf
import torch
from mup_transfer.architectures.mlp import mlp_constructor
import wandb

from mup_transfer.datasets.cifar10 import cifar10_constructor
from mup_transfer.config import ConfigBase
from mup_transfer.datasets.util import get_input_shape, get_output_size
from mup_transfer.loggers.wandb_logger import WandbLogger
from mup_transfer.mup.inf_types import infer_inf_type_sequential_model
from mup_transfer.mup.init import mup_initialise
from mup_transfer.mup.optim_params import get_mup_sgd_param_groups


@hydra.main(config_path="configs/", config_name="defaults")
def main(cfg: ConfigBase):
    """
    cfg is typed as ConfigBase for duck-typing, but during runtime it's actually an OmegaConf object.
    """
    # --- Runtime setup (logging directories, etc.)
    wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # --- Set up logging
    logger = WandbLogger(
        project_name="phenom-bayes",
        # This is needed to make WandB and Hydra play nicely:
        settings=wandb.Settings(start_method="thread"),
    )

    # --- Construct and get the dataset
    # TODO: Make general and dependent on the config
    train_dataset, eval_datasets = cifar10_constructor()

    # --- Construct the model

    # TODO: Make general and dependent on the config
    model = mlp_constructor(
        input_size=reduce(lambda x, y: x * y, get_input_shape(train_dataset)),
        hidden_sizes=[128, 128],
        output_size=get_output_size(train_dataset),
    )

    # Initialise the model with mup
    param_inf_types = infer_inf_type_sequential_model(model)
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
    model_forward = torch.compile(model)

    # --- Training loop

    # --- Save the final model

    # --- Final evaluations


if __name__ == "__main__":
    main()
