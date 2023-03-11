from dataclasses import dataclass


from mup_transfer.config_schemas import ArchitectureConfig, ArchitectureType, ConfigBase, DataLoaderConfig, DatasetConfig, DatasetType, InitialisationConfig, OptimizerConfig, OptimizerType

from hydra.core.config_store import ConfigStore


@dataclass
class TransformerArchitectureConfig(ArchitectureConfig):
    bias: bool
    d_model: int
    ffn_ratio: int
    nlayers: int
    nhead: int
    dropout: float
    tied: bool
    init_var: float
    ntokens: int
    att_mult: int
    output_mul: int
    # name: ArchitectureType = ArchitectureType.TRANSFORMER


class SequenceDataLoaderConfig(DataLoaderConfig):
    bptt: int = 35



# Specific experiment configs
def register_configs():
    # from mup_transfer.config_schemas import register_config
    cs = ConfigStore.instance()

    # Using the type
    cifar10_mlp_config = ConfigBase(
        architecture=ArchitectureConfig( name=ArchitectureType.MLP),
        dataset=DatasetConfig(name=DatasetType.CIFAR10),
        optimisation=OptimizerConfig(optimizer_type=OptimizerType.SGD),
        initialisation=InitialisationConfig(),
    )

    wikitext_transformer_config = ConfigBase(
        architecture=TransformerArchitectureConfig(
            bias=False,
            d_model=256,
            ffn_ratio=1,
            nlayers=2,
            nhead=2,
            dropout=0.2,
            tied=False,
            init_var=1,
            ntokens=33278,
            att_mult=1,
            output_mul=1,
            name=ArchitectureType.TRANSFORMER,
        ),
        dataset=DatasetConfig(name=DatasetType.WIKITEXT),
        optimisation=OptimizerConfig(optimizer_type=OptimizerType.ADAM),
        initialisation=InitialisationConfig(),
    )

    cs.store(name="cifar10_mlp", node=cifar10_mlp_config)
    cs.store(name="wikitext_transformer", node=wikitext_transformer_config)
