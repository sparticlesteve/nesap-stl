#!/bin/bash

module purge
module load cgpu
module load pytorch/v1.6.0-gpu

nTasks=1
gpusPerTask=2

srun -C gpu -n $nTasks --gpus-per-task $gpusPerTask -c 20 -t 30 -u \
    python train.py configs/predrnn3d_stlearn.yaml \
    -d nccl --gpus-per-rank $gpusPerTask -v
