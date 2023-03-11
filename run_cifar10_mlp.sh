#!/bin/bash
#SBATCH -p gpu-2080ti                       # partition
#SBATCH --gres=gpu:rtx2080ti:1              # type and number of gpus
#SBATCH --time=72:00:00                     # job will be cancelled after max. 72h
#SBATCH --output=cifar10_mlp_%A_%a.out
#SBATCH --array=4-14

# print info about current job
scontrol show job $SLURM_JOB_ID

export WANDB_API_KEY=9a337dffa0b41297c8870d83df890ea39dc02163

# Assuming you have activated your conda environment
for width in 64 128 256 512 1024 2048 4096; do
    python scripts/run.py \
        +experiment=cifar10_mlp \
        optimization.lr=1e-$SLURM_ARRAY_TASK_ID \
        mlp_config.hidden_sizes=\[$width,$width\]
done   
