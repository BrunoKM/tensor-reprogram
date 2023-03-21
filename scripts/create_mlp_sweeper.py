from typing import TypedDict
import numpy as np
import wandb
import hydra
import dataclasses
from hydra.core.config_store import ConfigStore


def get_mean_and_std_of_uniform(low, high):
    mu = (low + high) / 2
    sigma = (high - low) / (2 * np.sqrt(3))
    return {"mu": float(mu), "sigma": float(sigma)}



sweep_config = {
    "method": "bayes",
    "metric": {"name": "train_loss", "goal": "minimize"},
    "name": "sweep",
    "parameters": {
        "initialisation": {
            "parameters": {
            "init_scales_per_param": {
                f"{layer_num}.{param_name}": {
                    "distribution": "log_normal",
                    **get_mean_and_std_of_uniform(np.log(1e-3), np.log(1e2)),
                }
                for layer_num in [1, 3, 5]
                for param_name in ["weight", "bias"]
            },
            }
        },
        "optimization": {
            "parameters": {
            "per_param_lr": {
                f"{layer_num}.{param_name}": {
                    "distribution": "log_normal",
                    **get_mean_and_std_of_uniform(np.log(1e-10), np.log(1e3)),
                }
                for layer_num in [1, 3, 5]
                for param_name in ["weight", "bias"]
            },
            }
        },
    },
}


def main():
    import pprint
    pprint.pprint(sweep_config)
    import yaml
    # Convert nested dictionary to a yaml string
    yaml_string = yaml.dump(sweep_config)
    print(yaml_string)

    wandb.sweep(
        sweep=sweep_config,
        # project="cifar10-mlp-sweep",
        project="tensor-reprogram",
        entity="tensor-programs-v-reproduction",  # Log to the team's entity project
    )


if __name__ == "__main__":
    main()
