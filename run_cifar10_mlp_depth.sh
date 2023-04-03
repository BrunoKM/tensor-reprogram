#!/bin/bash
#SBATCH -p gpu-2080ti                       # partition
#SBATCH --gres=gpu:rtx2080ti:1              # type and number of gpus
#SBATCH --time=72:00:00                     # job will be cancelled after max. 72h
#SBATCH --output=cifar10_mlp_depth_%A_%a.out
#SBATCH --array=4-14

# Print info about current job.
scontrol show job $SLURM_JOB_ID

# Add your wandb api key here.
export WANDB_API_KEY=XXX

# Assuming you have activated your conda environment
for depth in 3 4 5 6; do
    python scripts/run.py \
        +experiment=cifar10_mlp \
        optimization.lr=$(perl -e "print 2**-$SLURM_ARRAY_TASK_ID") \
        mlp_config.width=256 \
        mlp_config.depth=$depth
done
