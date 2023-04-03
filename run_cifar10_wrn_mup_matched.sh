#!/bin/bash
#SBATCH -p gpu-2080ti                       # partition
#SBATCH --gres=gpu:rtx2080ti:1              # type and number of gpus
#SBATCH --time=72:00:00                     # job will be cancelled after max. 72h
#SBATCH --output=cifar10_wrn_mup_matched_%A_%a.out
#SBATCH --array=0-10

# Print info about current job.
scontrol show job $SLURM_JOB_ID

# Add your wandb api key here.
export WANDB_API_KEY=XXX

if [ $SLURM_ARRAY_TASK_ID == 0 ]
then
    GLOBAL_LR=$(perl -e "print 66e-4")
elif [ $SLURM_ARRAY_TASK_ID == 1 ]
then
    GLOBAL_LR=$(perl -e "print 1e-3")
elif [ $SLURM_ARRAY_TASK_ID == 2 ]
then
    GLOBAL_LR=$(perl -e "print 33e-3")
elif [ $SLURM_ARRAY_TASK_ID == 3 ]
then
    GLOBAL_LR=$(perl -e "print 66e-3")
elif [ $SLURM_ARRAY_TASK_ID == 4 ]
then
    GLOBAL_LR=$(perl -e "print 1e-2")
elif [ $SLURM_ARRAY_TASK_ID == 5 ]
then
    GLOBAL_LR=$(perl -e "print 33e-2")
elif [ $SLURM_ARRAY_TASK_ID == 6 ]
then
    GLOBAL_LR=$(perl -e "print 66e-2")
elif [ $SLURM_ARRAY_TASK_ID == 7 ]
then
    GLOBAL_LR=$(perl -e "print 1e-1")
elif [ $SLURM_ARRAY_TASK_ID == 8 ]
then
    GLOBAL_LR=$(perl -e "print 33e-1")
elif [ $SLURM_ARRAY_TASK_ID == 9 ]
then
    GLOBAL_LR=$(perl -e "print 66e-1")
elif [ $SLURM_ARRAY_TASK_ID == 10 ]
then
    GLOBAL_LR=$(perl -e "print 1e0")
fi

# Assuming you have activated your conda environment
for width_factor in 1 2 4 8; do
    python scripts/run.py \
        +experiment=cifar10_wrn \
        wrn_config.width_factor=$width_factor \
        optimization.global_lr=$GLOBAL_LR \
	++optimization.per_param_lr="{0.weight: 0.0625, 1.bias: 0.0625, 1.weight: 0.0625, 11.bias: 0.1, 11.weight: 64, 3.path2.0.bias: 0.0625, 3.path2.0.weight: 0.0625, 3.path2.2.weight: 1.0, 3.path2.3.bias: 0.0625, 3.path2.3.weight: 0.0625, 3.path2.5.weight: 1.0, 4.path2.0.bias: 0.0625, 4.path2.0.weight: 0.0625, 4.path2.2.weight: 1.0, 4.path2.3.bias: 0.0625, 4.path2.3.weight: 0.0625, 4.path2.5.weight: 1.0, 5.path1.weight: 1.0, 5.path2.0.bias: 0.0625, 5.path2.0.weight: 0.0625, 5.path2.2.weight: 1.0, 5.path2.3.bias: 0.03125, 5.path2.3.weight: 0.03125, 5.path2.5.weight: 1.0, 6.path2.0.bias: 0.03125, 6.path2.0.weight: 0.03125, 6.path2.2.weight: 1.0, 6.path2.3.bias: 0.03125, 6.path2.3.weight: 0.03125, 6.path2.5.weight: 1.0, 7.path1.weight: 1.0, 7.path2.0.bias: 0.03125, 7.path2.0.weight: 0.03125, 7.path2.2.weight: 1.0, 7.path2.3.bias: 0.015625, 7.path2.3.weight: 0.015625, 7.path2.5.weight: 1.0, 8.path2.0.bias: 0.015625, 8.path2.0.weight: 0.015625, 8.path2.2.weight: 1.0, 8.path2.3.bias: 0.015625, 8.path2.3.weight: 0.015625, 8.path2.5.weight: 1.0}" \
        ++initialisation.init_scales_per_param="{0.weight: 0.20412414523193154, 3.path2.2.weight: 0.47140452079103173, 3.path2.5.weight: 0.47140452079103173, 4.path2.2.weight: 0.47140452079103173, 4.path2.5.weight: 0.47140452079103173, 5.path1.weight: 1.0, 5.path2.2.weight: 0.33333333333333337, 5.path2.5.weight: 0.47140452079103184, 6.path2.2.weight: 0.47140452079103184, 6.path2.5.weight: 0.47140452079103184, 7.path1.weight: 1.0000000000000002, 7.path2.2.weight: 0.3333333333333334, 7.path2.5.weight: 0.47140452079103173, 8.path2.2.weight: 0.47140452079103173, 8.path2.5.weight: 0.47140452079103173, 11.weight: 4.618802153517007, 11.bias: 0.0721687836487}"
done
