"""Pytorch dataset specification for the LANL climate data

TODO: Finish a first working version is this implementation.
TODO: add support for non-overlapping samples (maybe the default).
TODO: move to pre-split train/val/test input files.
"""

# System
import logging

# Externals
import numpy as np
import torch

# Locals
from utils.preprocess import reshape_patch

class ClimateDataset(torch.utils.data.Dataset):
    """Climate dataset for spatio-temporal learning.

    Note that here we have changed the dataset from the original LANL folks'
    code to more closely match the auto-regressive style training of the
    PredRNN++ model. So this dataset doesn't provide a target value, but just
    returns a sequence. Then it is up to the Trainer and model to implement the
    desired training objective.

    FIXME: pred_len is not utilized correctly here. Maybe just remove it.
    TODO: add support for specifying non-overlapping samples.
    TODO: add PCA if needed.
    """

    def __init__(self, data, seq_len=36):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.shape[0] - self.seq_len + 1

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index : index + self.seq_len])
        return x

def get_datasets(data_file, n_train=8192, n_valid=2048, seq_len=36,
                 patch_size=1, **kwargs):
    """Factory function for the datasets.

    FIXME: assumes overlapping sequences in the train/val split.
    """
    with np.load(data_file) as f:
        data = f['temp'].astype(np.float32)

    # Reshape data into patches
    # The reshape function assumes shape (N, T, C, H, W)
    # but here the sample and sequence dims are merged: (T, H, W)
    # so we add a temporary dummy batch dim and channel dim: (1, T, 1, H, W)
    data = data[None, :, None, :, :]
    data = reshape_patch(data, patch_size)
    # Remove the dummy batch dim
    data = data[0]

    # Split into train/val
    n_train_seq = n_train + seq_len - 1
    train_data, valid_data = data[:n_train_seq], data[n_train_seq:]

    # Construct the PyTorch datasets
    train_set = ClimateDataset(train_data, seq_len=seq_len, **kwargs)
    valid_set = ClimateDataset(valid_data, seq_len=seq_len, **kwargs)
    return train_set, valid_set, {}
