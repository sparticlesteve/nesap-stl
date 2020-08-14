#!/bin/bash

module purge
module load esslurm
module load pytorch/v1.5.0-gpu

nTasks=1

srun -C gpu -n $nTasks --gpus-per-task 2 -c 20 -t 30 \
    python train.py configs/predrnn3d_stlearn.yaml --rank-gpu -v
