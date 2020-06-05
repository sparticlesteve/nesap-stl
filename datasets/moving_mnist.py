"""PyTorch dataset specification for the Moving MNIST dataset

The data comes from the PredRNN++ paper:
https://onedrive.live.com/?authkey=%21AGzXjcOlzTQw158&id=FF7F539F0073B9E2%21124&cid=FF7F539F0073B9E2
"""

# System
import os

# Externals
import numpy as np
import torch

# Locals
from utils.preprocess import reshape_patch

class MovingMNIST(torch.utils.data.Dataset):
    """Moving MNIST dataset"""

    def __init__(self, data_file, n_samples=None,
                 sample_shape=(20, 1, 64, 64), patch_size=4):
        self.data_file = data_file

        # Load the data
        with np.load(data_file) as f:
            d = f['input_raw_data']

        # Reshape and select requested number of samples
        d = d.reshape((-1,) + sample_shape)
        if n_samples is not None:
            d = d[:n_samples]

        # The original PredRNN++ code applies this patch transform which
        # breaks the image up into patch_size patches stacked as channels.
        d = reshape_patch(d, patch_size)

        # Convert to Torch tensor
        self.data = torch.tensor(d)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def get_datasets(data_dir, n_train=None, n_valid=None, **kwargs):
    data_dir = os.path.expandvars(data_dir)
    train_data = MovingMNIST(os.path.join(data_dir, 'moving-mnist-train.npz'),
                             n_samples=n_train, **kwargs)
    valid_data = MovingMNIST(os.path.join(data_dir, 'moving-mnist-valid.npz'),
                             n_samples=n_valid, **kwargs)
    return train_data, valid_data, {}
