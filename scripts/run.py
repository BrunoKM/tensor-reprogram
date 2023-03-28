import os
import math
import hydra
import omegaconf
import torch
import torch.nn as nn
import tqdm
import wandb
import pprint
import logging

from functools import reduce
from pathlib import Path
from mup_transfer.architectures.mlp import mlp_constructor
from hydra.core.config_store import ConfigStore
from mup_transfer.architectures.transformer import transformer_constructor
from mup_transfer.architectures.resnet import wide_resnet_constructor

from mup_transfer.data_utils import get_data_loaders
from mup_transfer.datasets.cifar10 import cifar10_constructor
from mup_transfer.config_schemas import ArchitectureType, ConfigBase, DatasetType, OptimizerType, ParameterisationType
from mup_transfer.datasets.util import get_input_shape, get_output_size
from mup_transfer.datasets.wikitext2 import wikitext_constructor
from mup_transfer.loggers.wandb_logger import WandbLogger
from mup_transfer.mup.inf_types import get_inf_types
from mup_transfer.mup.utils import get_param_name
from mup_transfer.mup.init import mup_initialise, standard_param_initialise, torch_param_initialise, scale_init_inplace
from mup_transfer.mup.optim_params import get_adam_param_groups, get_mup_sgd_param_groups
from mup_transfer.train import train
from mup_transfer.eval import eval
from mup_transfer.mup.inf_types import InfType

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
        project=config.wandb_project_name,
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
            Path(__file__).parent.parent
            / "data",  # Default data directory at the root of repository
            use_data_augmentation=config.dataset.use_data_augmentation,
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
            root=Path(__file__).parent.parent
            / "data",  # Default data directory at the root of repository
            train_batch_size=config.data_loader.train_batch_size,
            test_batch_size=config.data_loader.eval_batch_size,
            bptt=config.data_loader.bptt,
        )
    else:
        raise NotImplementedError(f"Dataset type {config.dataset_type} not implemented.")

    # --- Construct the model

    if config.architecture_type == ArchitectureType.MLP:
        # Avoid silent unintended behaviour.
        if (config.mlp_config.hidden_sizes is not None) == (
            config.mlp_config.width is not None and config.mlp_config.depth is not None
        ):
            raise ValueError(
                f"Either specify 'hidden_sizes' OR both 'width' and 'depth'.\n"
                f"Currenly: 'hidden_sizes'={config.mlp_config.hidden_sizes}, 'width'={config.mlp_config.width}, 'depth'={config.mlp_config.depth}."
            )
        if config.mlp_config.hidden_sizes is not None:
            hidden_sizes = config.mlp_config.hidden_sizes
        else:
            assert (
                config.mlp_config.width is not None and config.mlp_config.depth is not None
            ), "If hidden sizes not specified, specify both width and depth."
            hidden_sizes = [config.mlp_config.width for _ in range(config.mlp_config.depth)]

        model = mlp_constructor(
            input_size=reduce(lambda x, y: x * y, get_input_shape(train_loader.dataset)),
            hidden_sizes=hidden_sizes,
            output_size=get_output_size(train_loader.dataset),
            bias=config.mlp_config.add_bias,
        )
        # Get inf types for model
        param_inf_types = get_inf_types(
            model=model,
            input_weights_names=["input_layer.weight"],
            output_weights_names=["output_layer.weight"],
        )
    elif config.architecture_type == ArchitectureType.TRANSFORMER:
        model = transformer_constructor(config.transformer_config)
        param_inf_types = get_inf_types(
            model=model,
            input_weights_names=[
                get_param_name(
                    model,
                    # Get the weight of the first nn.Embedding layer in the model.
                    next(module for module in model.modules() if isinstance(module, nn.Embedding)).weight,  # type: ignore
                ),
            ],
            output_weights_names=[get_param_name(model, model.decoder.weight)],  # type: ignore
        )
    elif config.architecture_type == ArchitectureType.WRN:
        if config.wrn_config.blocks_per_stage is None:
            raise ValueError("Must specify blocks_per_stage for Wide-ResNet.")
        if config.wrn_config.width_factor is None:
            raise ValueError("Must specify width_factor for Wide-ResNet.")
        model = wide_resnet_constructor(
            blocks_per_stage=config.wrn_config.blocks_per_stage,
            width_factor=config.wrn_config.width_factor,
        )
        param_inf_types = get_inf_types(
            model=model,
            input_weights_names=[get_param_name(model, model[0].weight)],  # type: ignore
            output_weights_names=[
                get_param_name(
                    model,
                    # Get the weight of the first nn.Linear layer in the model.
                    next(module for module in model.modules() if isinstance(module, nn.Linear)).weight,  # type: ignore
                ),
            ],
        )
    else:
        raise ValueError(f"Unknown architecture type: {config.architecture_type}")

    model.to(DEVICE)

    # --- Initialise the model
    # Initialise the model with mup
    named_params = list(model.named_parameters())
    param_names = {name for name, _ in named_params}

    params_without_init = set()
    for module_name, module_type in model.named_modules():
        if isinstance(module_type, (torch.nn.LayerNorm, torch.nn.modules.batchnorm._BatchNorm)):
            logging.info(f"Module without mup initialization: {module_name} {module_type}")
            params_without_init.update({get_param_name(model, param) for param in module_type.parameters()})
    logging.info(f"Params without mup initialization: {params_without_init}")

    for param_name in config.initialisation.init_scales_per_param.keys():
        if param_name in params_without_init:
            raise ValueError(f"Initialize parameters of a LayerNorm/BatchNorm layer: {param_name}")

    init_scales = {
        name: (
            config.initialisation.init_scales_per_param[name]
            if name in config.initialisation.init_scales_per_param.keys()
            else config.initialisation.default_init_scale
        ) * config.initialisation.global_init_scale
        for name, _ in named_params
    }
    # Validate that all the init_scales_per_param parameter names are valid:
    for name in config.initialisation.init_scales_per_param.keys():
        if name not in param_names:
            raise ValueError(
                f"Parameter name '{name}' in 'init_scales_per_param' is not a valid parameter name."
                "\nValid parameter names are: {param_names}"
            )
    logging.info(f"Initialisation scales: {init_scales}")
    if config.parameterisation == ParameterisationType.MUP:
        mup_initialise(
            named_params=named_params,
            param_inf_types=param_inf_types,
            init_scale=init_scales,
            distribution=config.initialisation.init_distribution,
            params_without_init=params_without_init,
        )
    elif config.parameterisation == ParameterisationType.SP:
        # If not using muP, initialise using SP.
        standard_param_initialise(
            named_params=named_params,
            init_scale=init_scales,
            distribution=config.initialisation.init_distribution,
            params_without_init=params_without_init,
        )
    elif config.parameterisation == ParameterisationType.PYTORCH:
        torch_param_initialise(
            named_params=named_params,
            init_scale=init_scales,
            distribution=config.initialisation.init_distribution,
            params_without_init=params_without_init,
        )
    elif config.parameterisation == ParameterisationType.NONE:
        scale_init_inplace(named_params, init_scales)
    else:
        raise ValueError(f"Unknown parameterisation: {config.parameterisation}")

    def _calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            # math.prod is not always available, accumulate the product manually
            # we could use functools.reduce but that is not supported by TorchScript
            for s in tensor.shape[2:]:
                receptive_field_size *= s
        fan_in = num_input_fmaps #* receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    scales_dict = {}
    lrs_dict = {}
    for name, param in named_params:
        #if name in params_without_init:
        #    continue
        if param.dim() >= 2:
            fan_in, fan_out = _calculate_fan_in_and_fan_out(param)
        else:
            fan_in, fan_out = 64, 10  #_calculate_fan_in_and_fan_out(dict(*zip(named_params))[name.removesuffix('bias') + 'weight']) 
        inf_type = param_inf_types[name]
        match inf_type:
            case InfType.INPUT_OR_BIAS:
                if param.dim() < 2:
                    scale = 1.0 / fan_in ** 0.5 / math.sqrt(3)
                else:
                    scale = fan_in ** 0.5 * math.sqrt(2) / (fan_out ** 0.5)
                lr = 1 / param.shape[0]
            case InfType.HIDDEN_WEIGHT:
                scale = fan_in ** 0.5 * math.sqrt(2) / (fan_out ** 0.5)
                lr = 1.0
            case InfType.OUTPUT_WEIGHT:
                scale = fan_in ** 0.5 / math.sqrt(3)
                lr = fan_in
            case _:
                raise ValueError(f"Unrecognised infinite width type: {inf_type}")
        scales_dict[name] = scale
        lrs_dict[name] = lr
    logging.info('transformed scales dict')
    pprint.pprint(scales_dict)
    logging.info('transformed lrs dict')
    pprint.pprint(lrs_dict)
    logging.info('std dict:')
    std_dict = {name: param.std().item() for name, param in named_params}
    pprint.pprint(std_dict)

    # --- Construct the optimizer

    # Validate that all the per_param_lr parameter names are valid:
    for name in config.optimization.per_param_lr.keys():
        if name not in param_names:
            raise ValueError(
                f"Parameter name '{name}' in 'per_param_lr' is not a valid parameter name."
                "\nValid parameter names are: {param_names}"
            )
    # Learning rates per param:
    lr_scale_per_param = {
        name: (
            config.optimization.per_param_lr[name]
            if name in config.optimization.per_param_lr.keys()
            else config.optimization.default_lr
        ) * config.optimization.global_lr
        for name, param in named_params
    }
    logging.info(f"Base learning rates per parameter: {lr_scale_per_param}")

    if config.optimization.optimizer_type == OptimizerType.SGD:
        optim_constructor = torch.optim.SGD
    elif config.optimization.optimizer_type == OptimizerType.ADAM:
        optim_constructor = torch.optim.Adam
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimization.optimizer_type}")

    if config.parameterisation == ParameterisationType.MUP:
        if config.optimization.optimizer_type == OptimizerType.SGD:
            param_groups = get_mup_sgd_param_groups(
                named_params=named_params,
                init_lr_scale=lr_scale_per_param,
                param_inf_types=param_inf_types,
            )
        elif config.optimization.optimizer_type == OptimizerType.ADAM:
            param_groups = get_adam_param_groups(
                named_params=named_params,
                init_lr_scale=lr_scale_per_param,
                param_inf_types=param_inf_types,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {config.optimization.optimizer_type}")
    else:
        param_groups = [
            {"params": [param], "lr": lr_scale_per_param[name]}
            for name, param in model.named_parameters()
        ]
    for group in param_groups:
        lr = group['lr']
        print(f'effective lr: {lr}')

    optim = optim_constructor(
        params=param_groups,  # type: ignore
        lr=config.optimization.default_lr,
        **config.optimization.optimizer_kwargs,
    )

    if config.optimization.cosine_lr_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=config.num_epochs * len(train_loader),
        )
    else:
        scheduler = None

    # --- Compile the model
    # model = torch.compile(model)

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
            scheduler=scheduler,
            device=DEVICE,
            logger=logger,
        )
        logger.log_scalar("train.epoch_loss", epoch_loss)
        logger.log_scalar("train.epoch_accuracy", epoch_accuracy)
        if float(epoch_loss) == float("nan"):
            raise ValueError("Training loss is NaN")
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
