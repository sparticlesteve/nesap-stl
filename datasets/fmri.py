"""PyTorch dataset specification for the fMRI dataset"""

# System
import os

# Externals
import numpy as np
import torch

# Locals
from utils.preprocess import reshape_patch_3d

class FMRIDataset(torch.utils.data.Dataset):
    """PyTorch dataset for the resting-stage fMRI"""

    def __init__(self, data_files,
                 image_crop=((13, 13), (13, 13), (0, 0)),
                 image_padding=((0, 0), (0, 0), (2, 2)),
                 time_frames=32, patch_size=1):
        self.data_files = data_files
        self.image_crop = image_crop
        self.padding = image_padding + ((0, 0),)
        self.time_frames = time_frames
        self.patch_size = patch_size

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        x = np.load(self.data_files[index])

        # Apply cropping
        xcrop, ycrop, zcrop = self.image_crop
        x = x[xcrop[0] : x.shape[0] - xcrop[1],
              ycrop[0] : x.shape[1] - ycrop[1],
              zcrop[0] : x.shape[2] - zcrop[1],
              0 : self.time_frames]

        # Apply padding
        x = np.pad(x, self.padding)

        # Change format (H,W,D,T) -> (T,H,W,D)
        x = x.transpose(3, 0, 1, 2)

        # Split into patches (briefly insert dummy batch dim)
        x = reshape_patch_3d(x[None, :, None], self.patch_size).squeeze(0)

        return torch.from_numpy(x)

def get_datasets(data_dir, n_train, n_valid, **kwargs):

    # Get the list of files
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    train_files = data_files[:n_train]
    valid_files = data_files[n_train:n_train+n_valid]

    train_data = FMRIDataset(train_files, **kwargs)
    valid_data = FMRIDataset(valid_files, **kwargs)
    return train_data, valid_data, {}

def _test():
    data_dir = '/global/cscratch1/sd/yanzhang/data_brain/fmri_numpy_380'
    get_datasets(data_dir, 128, 16, patch_size=2)

if __name__ == '__main__':
    _test()
