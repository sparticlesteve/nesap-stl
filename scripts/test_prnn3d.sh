#!/bin/bash

module purge
module load cgpu
module load pytorch/v1.6.0-gpu

# Debugging data-parallelism
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=COLL

nTasks=4
gpusPerTask=2
tasksPerNode=4

srun -C gpu -n $nTasks --ntasks-per-node $tasksPerNode \
    --gpus-per-task $gpusPerTask -c 20 -t 30 -u -l \
    python train.py configs/predrnn3d_stlearn.yaml \
    -d nccl --gpus-per-rank $gpusPerTask -v
