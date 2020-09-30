__author__ = 'yunbo'

import numpy as np

def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    batch_size = np.shape(img_tensor)[0]
    seq_length = np.shape(img_tensor)[1]
    num_channels = np.shape(img_tensor)[2]
    img_height = np.shape(img_tensor)[3]
    img_width = np.shape(img_tensor)[4]
    a = np.reshape(img_tensor, [batch_size, seq_length, num_channels,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size])
    b = np.transpose(a, [0,1,2,4,6,3,5])
    patch_tensor = np.reshape(b, [batch_size, seq_length,
                                  num_channels*patch_size*patch_size,
                                  img_height//patch_size,
                                  img_width//patch_size])
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    assert 5 == patch_tensor.ndim
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    channels = np.shape(patch_tensor)[2]
    patch_height = np.shape(patch_tensor)[3]
    patch_width = np.shape(patch_tensor)[4]
    img_channels = channels / (patch_size*patch_size)
    a = np.reshape(patch_tensor, [batch_size, seq_length, img_channels,
                                  patch_size, patch_size,
                                  patch_height, patch_width])
    # FIXME: Is this really correct? Why isn't it [0,1,2,5,3,6,4]?
    b = np.transpose(a, [0,1,2,5,4,6,3])
    img_tensor = np.reshape(b, [batch_size, seq_length, img_channels,
                                patch_height * patch_size,
                                patch_width * patch_size])
    return img_tensor

def reshape_patch_3d(img_tensor, patch_size):
    batch_size, seq_length, num_channels, img_height, img_width, img_depth = img_tensor.shape
    a = np.reshape(img_tensor, [batch_size, seq_length, num_channels,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size,
                                img_depth//patch_size, patch_size])
    b = np.transpose(a, [0,1,2,4,6,8,3,5,7])
    patch_tensor = np.reshape(b, [batch_size, seq_length,
                                  num_channels*patch_size*patch_size*patch_size,
                                  img_height//patch_size,
                                  img_width//patch_size,
                                  img_depth//patch_size])
    return patch_tensor

def reshape_patch_back_3d(patch_tensor, patch_size):
    batch_size, seq_length, channels, patch_height, patch_width, patch_depth = patch_tensor.shape
    img_channels = channels / (patch_size*patch_size*patch_size)
    a = np.reshape(patch_tensor, [batch_size, seq_length, img_channels,
                                  patch_size, patch_size, patch_size,
                                  patch_height, patch_width, patch_depth])
    # TODO: validate this ordering
    b = np.transpose(a, [0,1,2,6,3,7,4,8,5])
    img_tensor = np.reshape(b, [batch_size, seq_length, img_channels,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                patch_depth * patch_size])
    return img_tensor
