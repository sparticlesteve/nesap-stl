# NESAP Extreme Spatio-Temporal Learning

Deep learning on large spatio-temporal data, including fMRI and climate data.

## Datasets

- Moving MNIST
- Brain fMRI
- Climate

## Models

- PredRNN++

## Package layout

The directory layout of this repo is designed to be flexible:
- Configuration files (in YAML format) go in `configs/`
- Dataset specifications using PyTorch's Dataset API go into `datasets/`
- Model implementations go into `models/`
- Trainer implementations go into `trainers/`. Trainers inherit from
  `BaseTrainer` and are responsible for constructing models as well as training
  and evaluating them.

All examples are run with the generic training script, `train.py`.

## How to run

To run the examples on the Cori supercomputer, you may use the provided
example SLURM batch script. Here's how to run the Hello World example on 4
Haswell nodes:

`sbatch -N 4 scripts/train_cori.sh configs/hello.yaml`
