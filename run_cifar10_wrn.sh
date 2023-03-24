#!/bin/bash
#SBATCH -p gpu-2080ti                       # partition
#SBATCH --gres=gpu:rtx2080ti:1              # type and number of gpus
#SBATCH --time=72:00:00                     # job will be cancelled after max. 72h
#SBATCH --output=cifar10_wrn_%A_%a.out
#SBATCH --array=0-10

# print info about current job
scontrol show job $SLURM_JOB_ID

export WANDB_API_KEY=9a337dffa0b41297c8870d83df890ea39dc02163

if [ $SLURM_ARRAY_TASK_ID == 0 ]
then
    GLOBAL_LR=$(perl -e "print 1e-4")
elif [ $SLURM_ARRAY_TASK_ID == 1 ]
then
    GLOBAL_LR=$(perl -e "print 5e-4")
elif [ $SLURM_ARRAY_TASK_ID == 2 ]
then
    GLOBAL_LR=$(perl -e "print 1e-3")
elif [ $SLURM_ARRAY_TASK_ID == 3 ]
then
    GLOBAL_LR=$(perl -e "print 5e-3")
elif [ $SLURM_ARRAY_TASK_ID == 4 ]
then
    GLOBAL_LR=$(perl -e "print 1e-2")
elif [ $SLURM_ARRAY_TASK_ID == 5 ]
then
    GLOBAL_LR=$(perl -e "print 5e-2")
elif [ $SLURM_ARRAY_TASK_ID == 6 ]
then
    GLOBAL_LR=$(perl -e "print 1e-1")
elif [ $SLURM_ARRAY_TASK_ID == 7 ]
then
    GLOBAL_LR=$(perl -e "print 5e-1")
elif [ $SLURM_ARRAY_TASK_ID == 8 ]
then
    GLOBAL_LR=$(perl -e "print 1e0")
elif [ $SLURM_ARRAY_TASK_ID == 9 ]
then
    GLOBAL_LR=$(perl -e "print 5e0")
elif [ $SLURM_ARRAY_TASK_ID == 10 ]
then
    GLOBAL_LR=$(perl -e "print 1e1")
fi

# Assuming you have activated your conda environment
for width_factor in 1 2 4 8; do
    python scripts/run.py \
        +experiment=cifar10_wrn \
        wrn_config.width_factor=$width_factor \
        optimization.default_lr=$GLOBAL_LR \
done