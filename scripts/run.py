import hydra
import omegaconf
import torch
import wandb

from functools import reduce
from pathlib import Path
from mup_transfer.architectures.mlp import mlp_constructor
from hydra.core.config_store import ConfigStore

from mup_transfer.data_utils import get_data_loaders
from mup_transfer.datasets.cifar10 import cifar10_constructor
from mup_transfer.config import ConfigBase
from mup_transfer.datasets.util import get_input_shape, get_output_size
from mup_transfer.loggers.wandb_logger import WandbLogger
from mup_transfer.mup.inf_types import get_inf_types, infer_inf_type_sequential_model
from mup_transfer.mup.utils import get_param_name
from mup_transfer.mup.init import mup_initialise
from mup_transfer.mup.optim_params import get_mup_sgd_param_groups
from mup_transfer.train import train
from mup_transfer.eval import eval

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        Path(__file__).parent.parent / "data",  # Default data directory at the root of repository
    )
    train_loader, eval_loaders = get_data_loaders(
        train_dataset,
        eval_datasets,
        train_batch_size=cfg.data_loader.train_batch_size,
        eval_batch_size=cfg.data_loader.eval_batch_size,
        num_workers=cfg.data_loader.num_workers,
        pin_memory=cfg.data_loader.pin_memory,
    )

    # --- Construct the model

    # TODO: Make general and dependent on the config
    model = mlp_constructor(
        input_size=reduce(lambda x, y: x * y, get_input_shape(train_dataset)),
        hidden_sizes=[128, 128],
        output_size=get_output_size(train_dataset),
    )
    model.to(DEVICE)

    # Initialise the model with mup
    param_inf_types = get_inf_types(
        model=model,
        input_weights_names=[get_param_name(model, model[0].weight)],  # type: ignore
        output_weights_names=[get_param_name(model, model[-1].weight)],  # type: ignore
    )
    mup_initialise(
        named_params=(named_params := list(model.named_parameters())),
        param_inf_types=param_inf_types,
        init_scale=cfg.initialisation.init_scale,
    )
    param_groups = get_mup_sgd_param_groups(
        named_params=named_params,
        init_lr_scale=cfg.optimisation.lr,
        param_inf_types=param_inf_types,
    )


    # --- Construct the optimizer
    # TODO make optim type configurable via config.
    optim = torch.optim.SGD(
        params=param_groups,  # type: ignore
        lr=cfg.optimisation.lr,
    )
    # TODO: Maybe add lr schedule.

    # --- Compile the model
    # model_forward = torch.compile(model)

    # --- Training and evaluation loop
    for _ in range(cfg.num_epochs):
        train_loss, train_accuracy = train(model, train_loader, optim, DEVICE)
        logger.log_scalar("train.loss", train_loss)
        logger.log_scalar("train.accuracy", train_accuracy)
        logger.increment_step()

        for eval_dataset_name, eval_loader in eval_loaders.items():
            eval_loss, eval_accuracy = eval(model, eval_loader, DEVICE)
            logger.log_scalar(f"{eval_dataset_name}.loss", eval_loss)
            logger.log_scalar("train.accuracy", eval_accuracy)

    # --- Save the final model
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    file_name = "test_run"  # TODO: Decide on naming scheme.
    torch.save(
        model_to_save.state_dict(),
        Path(__file__).parent.parent / f"trained_models/{file_name}",  # Default directory at the root of repository for trained models
    )


if __name__ == "__main__":
    main()
