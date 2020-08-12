import torch
import torch.nn as nn


class GHUBase(nn.Module):
    def __init__(self, filter_size, num_channels,
                 layer_norm=True, init_val=0.001):
        """Initialize the Gradient Highway Unit.
        """
        super(GHUBase, self).__init__()

        self.filter_size = filter_size
        self.num_channels = num_channels
        self.layer_norm = layer_norm

        self.conv_z = None
        self.conv_x = None

        self.ln_x = nn.LayerNorm(num_channels * 2, elementwise_affine=True)
        self.ln_z = nn.LayerNorm(num_channels * 2, elementwise_affine=True)

        self.init_val = init_val if init_val != -1 else None

    def init_conv(self):
        if not self.init_val:
            return

        nn.init.uniform_(self.conv_z.weight, -self.init_val, self.init_val)
        nn.init.uniform_(self.conv_x.weight, -self.init_val, self.init_val)

    def init_state(self, inputs, num_channels):
        dims = inputs.shape

        if len(dims) == self.num_dims + 2:
            batch, rest = dims[0], dims[2:]
        else:
            raise ValueError('input tensor should be rank {}.'.format(
                self.num_dims + 2))

        return torch.zeros([batch, num_channels, *rest],
                           dtype=inputs.dtype, device=inputs.device)

    def run_layer_norm(self, x, ln):
        idx = list(range(self.num_dims + 2))
        return ln(x.permute(0, *idx[2:], 1)).permute(0, -1, *idx[1:-1])

    def forward(self, x, z=None):
        if z is None:
            z = self.init_state(x, self.num_channels)

        z_concat = self.conv_z(z)
        x_concat = self.conv_x(x)

        if self.layer_norm:
            x_concat = self.run_layer_norm(x_concat, self.ln_x)
            z_concat = self.run_layer_norm(z_concat, self.ln_z)

        gates = x_concat + z_concat

        # into 2 parts at axis=1
        p, u = torch.split(gates, self.num_channels, 1)

        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1-u) * z

        return z_new


class GHU2d(GHUBase):
    def __init__(self, filter_size, num_channels,
                 layer_norm=False, init_val=0.001):
        super(GHU2d, self).__init__(
            filter_size, num_channels, layer_norm, init_val)

        self.num_dims = 2

        self.conv_z = nn.Conv2d(
            num_channels, num_channels * 2, filter_size, 1,
            filter_size // 2, padding_mode='replicate')
        self.conv_x = nn.Conv2d(
            num_channels, num_channels * 2, filter_size, 1,
            filter_size // 2, padding_mode='replicate')

        self.init_conv()


class GHU3d(GHUBase):
    def __init__(self, filter_size, num_channels,
                 layer_norm=False, init_val=0.001):
        super(GHU3d, self).__init__(
            filter_size, num_channels, layer_norm, init_val)

        self.num_dims = 3

        self.conv_z = nn.Conv3d(
            num_channels, num_channels * 2, filter_size, 1,
            filter_size // 2, padding_mode='replicate')
        self.conv_x = nn.Conv3d(
            num_channels, num_channels * 2, filter_size, 1,
            filter_size // 2, padding_mode='replicate')

        self.init_conv()
