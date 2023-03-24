#!/bin/bash
#SBATCH -p gpu-2080ti                       # partition
#SBATCH --gres=gpu:rtx2080ti:1              # type and number of gpus
#SBATCH --time=72:00:00                     # job will be cancelled after max. 72h
#SBATCH --output=cifar10_mlp_sp_%A_%a.out
#SBATCH --array=0-10

# print info about current job
scontrol show job $SLURM_JOB_ID

export WANDB_API_KEY=9a337dffa0b41297c8870d83df890ea39dc02163

if [ $SLURM_ARRAY_TASK_ID == 0 ]
then
    GLOBAL_LR=$(perl -e "print 1e-3")
elif [ $SLURM_ARRAY_TASK_ID == 1 ]
then
    GLOBAL_LR=$(perl -e "print 5e-3")
elif [ $SLURM_ARRAY_TASK_ID == 2 ]
then
    GLOBAL_LR=$(perl -e "print 1e-2")
elif [ $SLURM_ARRAY_TASK_ID == 3 ]
then
    GLOBAL_LR=$(perl -e "print 5e-2")
elif [ $SLURM_ARRAY_TASK_ID == 4 ]
then
    GLOBAL_LR=$(perl -e "print 1e-1")
elif [ $SLURM_ARRAY_TASK_ID == 5 ]
then
    GLOBAL_LR=$(perl -e "print 5e-1")
elif [ $SLURM_ARRAY_TASK_ID == 6 ]
then
    GLOBAL_LR=$(perl -e "print 1e0")
elif [ $SLURM_ARRAY_TASK_ID == 7 ]
then
    GLOBAL_LR=$(perl -e "print 5e0")
elif [ $SLURM_ARRAY_TASK_ID == 8 ]
then
    GLOBAL_LR=$(perl -e "print 1e1")
elif [ $SLURM_ARRAY_TASK_ID == 9 ]
then
    GLOBAL_LR=$(perl -e "print 5e1")
elif [ $SLURM_ARRAY_TASK_ID == 10 ]
then
    GLOBAL_LR=$(perl -e "print 1e2")
fi

# Assuming you have activated your conda environment
for width in 64 128 256 512 1024 2048 4096; do
    python scripts/run.py \
        +experiment=cifar10_mlp \
        parameterisation="SP" \
        initialisation.init_distribution="NORMAL" \
        optimization.global_lr=$GLOBAL_LR \
        mlp_config.hidden_sizes=\[$width,$width\] \
        ++optimization.per_param_lr="{hidden_layer0.bias: 16.6007237198, hidden_layer0.weight: 0.02152539891931283, input_layer.bias: 7.56926283168, input_layer.weight: 8.96819147457, output_layer.weight: 3.68129559e-7}" \
        ++initialisation.init_scales_per_param="{hidden_layer0.bias: 1.0, hidden_layer0.weight: 1.0, input_layer.bias: 1.0, input_layer.weight: 1.0, output_layer.weight: 0.0625}"
done
