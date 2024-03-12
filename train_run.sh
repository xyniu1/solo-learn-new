#!/bin/bash

#SBATCH --array=0
#SBATCH --job-name=bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=96:00:00
#SBATCH --constraint=a100
#SBATCH --output=bash_%a.out

# module load cuda/12 cudnn
module load slurm
# source /mnt/home/xniu1/miniconda3/etc/profile.d/conda.sh
# conda activate solo_new
source /mnt/home/xniu1/venvs/solo_new/bin/activate

master_node=$SLURMD_NODENAME

# readarray -t ALPHA < alpha.txt
# Alpha=${ALPHA[$SLURM_ARRAY_TASK_ID]}
# export Alpha

srun python `which torchrun` \
        --nnodes $SLURM_JOB_NUM_NODES \
        --nproc_per_node $SLURM_GPUS_PER_NODE \
        --rdzv_id $SLURM_JOB_ID \
        --rdzv_backend c10d \
        --rdzv_endpoint $master_node:29401 \
        main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name straight.yaml



