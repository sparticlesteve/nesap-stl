#!/bin/bash

module purge
module load esslurm
module load pytorch/v1.5.0-gpu

srun -C gpu -G 1 -c 10 -t 1:00:00 \
    python train.py configs/predrnn3d_test.yaml --rank-gpu -v
