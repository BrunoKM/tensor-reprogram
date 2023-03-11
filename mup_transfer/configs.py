from dataclasses import dataclass


from mup_transfer.config_schemas import ArchitectureType, ConfigBase, DataLoaderConfig, DatasetConfig, DatasetType, InitialisationConfig, OptimizerConfig, OptimizerType

from hydra.core.config_store import ConfigStore




# Specific experiment configs
def register_configs():
    cs = ConfigStore.instance()

    # Using the type
    cifar10_mlp_config = ConfigBase(
        architecture_type=ArchitectureType.MLP,
        dataset_type=DatasetType.CIFAR10,
        optimisation=OptimizerConfig(optimizer_type=OptimizerType.SGD),
        initialisation=InitialisationConfig(),
    )

    wikitext_transformer_config = ConfigBase(
        architecture_type=ArchitectureType.TRANSFORMER,
        dataset_type=DatasetType.WIKITEXT,
        optimisation=OptimizerConfig(optimizer_type=OptimizerType.ADAM),
        initialisation=InitialisationConfig(),
    )

    cs.store(name="cifar10_mlp", node=cifar10_mlp_config)
    cs.store(name="wikitext_transformer", node=wikitext_transformer_config)
