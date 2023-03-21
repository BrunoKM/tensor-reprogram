import argparse
import os
import json


def main(file_path):
    file = open(file_path, "r")
    python_dict = json.load(file)
    # Assumes specific structure of json file.
    per_param_lr = python_dict["optimization"]["per_param_lr"]
    init_scales_per_param = python_dict["initialisation"]["init_scales_per_param"]
    config_string = (f"++optimization.per_param_lr='{per_param_lr}' "
                     + f"++initialisation.init_scales_per_param='{init_scales_per_param}'")
    cmd_string = f"python3 scripts/run.py +experiment=cifar10_mlp mlp_config.hidden_sizes=\[256,256\] {config_string}"
    os.system(cmd_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path",
                        type=str,
                        help="Path to sweep config json file.")
    args = parser.parse_args()
    main(args.file_path)
