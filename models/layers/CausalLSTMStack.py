import torch
from torch import nn

from . import CausalLSTMCell2d, CausalLSTMCell3d, GHU2d, GHU3d

class CausalLSTMStack(nn.Module):
    def __init__(self,
                 filter_size,
                 num_dims,
                 channels,
                 layer_norm=True,
                 ):
        super(CausalLSTMStack, self).__init__()

        self.filter_size = filter_size
        self.num_dims = num_dims
        self.channels = channels
        self.num_layers = len(channels)

        assert self.num_layers >= 2

        if num_dims == 2:
            clstmc, ghu = CausalLSTMCell2d, GHU2d
        elif num_dims == 3:
            clstmc, ghu = CausalLSTMCell3d, GHU3d
        else:
            raise ValueError()

        self.lstms = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                n_hid_in, n_hid_out = channels[-1], channels[0]
            else:
                n_hid_in, n_hid_out = channels[i-1:i+1]

            cell = clstmc(filter_size, n_hid_in, n_hid_out,
                          layer_norm=layer_norm)

            self.lstms.append(cell)

        self.ghu = ghu(filter_size, channels[0], layer_norm=layer_norm)


    def forward(self, x, h_prev=None, c_prev=None, m_prev=None, z_prev=None):
        if h_prev is None:
            h_prev = [None] * self.num_layers
        if c_prev is None:
            c_prev = [None] * self.num_layers

        h_new = [None] * self.num_layers
        c_new = [None] * self.num_layers

        h, c, m = self.lstms[0](x, h_prev[0], c_prev[0], m_prev)
        h_new[0], c_new[0] = h, c

        z = self.ghu(h, z_prev)

        h, c, m = self.lstms[1](z, h_prev[1], c_prev[1], m)
        h_new[1], c_new[1] = h, c

        for k in range(2, self.num_layers):
            h, c, m = self.lstms[k](h_new[k - 1], h_prev[k], c_prev[k], m)
            h_new[k], c_new[k] = h, c

        return h_new, c_new, m, z
