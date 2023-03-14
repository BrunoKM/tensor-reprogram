import os
import hydra
import omegaconf
import torch
import torch.nn as nn
import tqdm
import wandb
import logging

from functools import reduce
from pathlib import Path
from mup_transfer.architectures.mlp import mlp_constructor
from hydra.core.config_store import ConfigStore
from mup_transfer.architectures.transformer import transformer_constructor

from mup_transfer.data_utils import get_data_loaders
from mup_transfer.datasets.cifar10 import cifar10_constructor
from mup_transfer.config_schemas import ArchitectureType, ConfigBase, DatasetType, OptimizerType
from mup_transfer.datasets.util import get_input_shape, get_output_size
from mup_transfer.datasets.wikitext2 import wikitext_constructor
from mup_transfer.loggers.wandb_logger import WandbLogger
from mup_transfer.mup.inf_types import get_inf_types
from mup_transfer.mup.utils import get_param_name
from mup_transfer.mup.init import mup_initialise
from mup_transfer.mup.optim_params import get_adam_param_groups, get_mup_sgd_param_groups
from mup_transfer.train import train
from mup_transfer.eval import eval

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Register the defaults from the structured dataclass config schema:
cs = ConfigStore.instance()
cs.store(name="config_base", node=ConfigBase)


@hydra.main(config_path="configs/", config_name="defaults", version_base=None)
def main(config: ConfigBase):
    """
    cfg is typed as ConfigBase for duck-typing, but during runtime it's actually an OmegaConf object.
    """
    logging.info(f"Hydra current working directory: {os.getcwd()}")
    # --- Runtime setup (logging directories, etc.)

    # --- Set up logging
    logger = WandbLogger(
        project="tensor-reprogram",
        entity="tensor-programs-v-reproduction",  # Log to the team's entity project
        # This is needed to make WandB and Hydra play nicely:
        settings=wandb.Settings(start_method="thread"),
        # Log the config to WandB
        config=omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        # Allow for disabling upload when testing code
        mode=("disabled" if not config.log_to_wandb else "online"),
    )

    # --- Construct and get the dataset
    if config.dataset_type == DatasetType.CIFAR10:
        train_dataset, eval_datasets = cifar10_constructor(
            Path(__file__).parent.parent / "data",  # Default data directory at the root of repository
        )
        train_loader, eval_loaders = get_data_loaders(
            train_dataset,
            eval_datasets,
            train_batch_size=config.data_loader.train_batch_size,
            eval_batch_size=config.data_loader.eval_batch_size,
            num_workers=config.data_loader.num_workers,
            pin_memory=config.data_loader.pin_memory,
        )
    elif config.dataset_type == DatasetType.WIKITEXT:
        train_loader, eval_loaders = wikitext_constructor(
            root=Path(__file__).parent.parent / "data",  # Default data directory at the root of repository
            train_batch_size=config.data_loader.train_batch_size,
            test_batch_size=config.data_loader.eval_batch_size,
            bptt=config.data_loader.bptt,
        )
    else:
        raise NotImplementedError(f"Dataset type {config.dataset_type} not implemented.")
    

    # --- Construct the model

    if config.architecture_type == ArchitectureType.MLP:
        # Avoid silent unintended behaviour.
        if (config.mlp_config.hidden_sizes is not None) != (config.mlp_config.width is not None and config.mlp_config.depth is not None):
            raise ValueError(
                f"Either specify 'hidden_sizes' OR both 'width' and 'depth'.\n"
                f"Currenly: 'hidden_sizes'={config.mlp_config.hidden_sizes}, 'width'={config.mlp_config.width}, 'depth'={config.mlp_config.depth}."
            )
        if config.mlp_config.hidden_sizes is not None:
            hidden_sizes = config.mlp_config.hidden_sizes
        else:
            assert config.mlp_config.width is not None and config.mlp_config.depth is not None, "If hidden sizes not specified, specify both width and depth."
            hidden_sizes = [config.mlp_config.width for _ in range(config.mlp_config.depth)]

        model = mlp_constructor(
            input_size=reduce(lambda x, y: x * y, get_input_shape(train_loader.dataset)),
            hidden_sizes=hidden_sizes,
            output_size=get_output_size(train_loader.dataset),
        )
        # Get inf types for model
        param_inf_types = get_inf_types(
            model=model,
            input_weights_names=[
                get_param_name(
                    model,
                    # Get the weight of the first nn.Linear layer in the model.
                    next(module for module in model if isinstance(module, nn.Linear)).weight, # type: ignore
                ),
            ],
            output_weights_names=[get_param_name(model, model[-1].weight)],  # type: ignore
        )
    elif config.architecture_type == ArchitectureType.TRANSFORMER:
        model = transformer_constructor(config.transformer_config)
        param_inf_types = get_inf_types(
            model=model,
            input_weights_names=[
                get_param_name(
                    model,
                    # Get the weight of the first nn.Linear layer in the model.
                    next(module for module in model.modules() if isinstance(module, nn.Embedding)).weight, # type: ignore
                ),
            ],
            output_weights_names=[get_param_name(model, model.decoder.weight)],  # type: ignore
        )
    else:
        raise ValueError(f"Unknown architecture type: {config.architecture_type}")

    model.to(DEVICE)

    # Initialise the model with mup
    named_params = list(model.named_parameters())
    if config.use_mu_param:
        mup_initialise(
            named_params=named_params,
            param_inf_types=param_inf_types,
            init_scale=config.initialisation.init_scale,
        )


    # --- Construct the optimizer
    if config.optimization.optimizer_type == OptimizerType.SGD:
        if config.use_mu_param:
            param_groups = get_mup_sgd_param_groups(
                named_params=named_params,
                init_lr_scale=config.optimization.lr,
                param_inf_types=param_inf_types,
            )
        else:
            param_groups = model.parameters()
        optim = torch.optim.SGD(
            params=param_groups,  # type: ignore
            lr=config.optimization.lr,
            **config.optimization.optimizer_kwargs,
        )
    elif config.optimization.optimizer_type == OptimizerType.ADAM:
        if config.use_mu_param:
            param_groups = get_adam_param_groups(
                named_params=named_params,
                init_lr_scale=config.optimization.lr,
                param_inf_types=param_inf_types,
            )
        else:
            param_groups = model.parameters()
        optim = torch.optim.Adam(
            params=param_groups,  # type: ignore
            lr=config.optimization.lr,
            **config.optimization.optimizer_kwargs,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimization.optimizer_type}")

    # TODO: Maybe add lr schedule.

    # --- Compile the model
    # model_forward = torch.compile(model)

    # --- Training and evaluation loop
    def eval_and_log():
        for eval_dataset_name, eval_loader in eval_loaders.items():
            eval_loss, eval_accuracy = eval(model, eval_loader, DEVICE)
            logger.log_scalar(f"{eval_dataset_name}.loss", eval_loss)
            logger.log_scalar(f"{eval_dataset_name}.accuracy", eval_accuracy)
        eval_loss, eval_accuracy = eval(model, train_loader, DEVICE)
        logger.log_scalar(f"train.loss", eval_loss)
        logger.log_scalar(f"train.accuracy", eval_accuracy)

    eval_and_log()
    for _ in tqdm.tqdm(range(config.num_epochs), desc="Training epochs"):
        epoch_loss, epoch_accuracy = train(
            model=model,
            train_loader=train_loader,
            optim=optim,
            clip_grad=config.optimization.clip_grad,
            device=DEVICE,
            logger=logger,
        )
        logger.log_scalar("train.epoch_loss", epoch_loss)
        logger.log_scalar("train.epoch_accuracy", epoch_accuracy)
        eval_and_log()

    # --- Save the final model
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(
        model_to_save.state_dict(),
        # Save the the working directory, which should be configured by Hydra to
        # be the output directory.
        Path(".") / f"model.torch",
    )


if __name__ == "__main__":
    main()
