import torch
from torch import nn


class CausalLSTMCellBase(nn.Module):
    def __init__(self, filter_size, in_channels, out_channels,
                 forget_bias=1.0, layer_norm=True, init_val=0.001):
        """
        Initialize the Causal LSTM cell.

        Parameters
        ==========
        filter_size:
            int tuple thats the height and width of the filter.
        in_channels:
            number of units for input tensor.
        out_channels:
            number of units for output tensor.
        seq_shape:
            shape of a sequence.
        forget_bias: float
            The bias added to forget gates.
        layer_norm:
            whether to apply tensor layer normalization
        """
        super(CausalLSTMCellBase, self).__init__()

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer_norm = layer_norm
        self.forget_bias = forget_bias

        self.conv_h = None
        self.conv_c = None
        self.conv_m = None
        self.conv_x = None
        self.conv_c2 = None
        self.conv_o = None
        self.conv_h2 = None

        self.ln_h = nn.LayerNorm(out_channels * 4, elementwise_affine=True)
        self.ln_c = nn.LayerNorm(out_channels * 3, elementwise_affine=True)
        self.ln_m = nn.LayerNorm(out_channels * 3, elementwise_affine=True)
        self.ln_x = nn.LayerNorm(out_channels * 7, elementwise_affine=True)
        self.ln_c2 = nn.LayerNorm(out_channels * 4, elementwise_affine=True)
        self.ln_o = nn.LayerNorm(out_channels, elementwise_affine=True)

        self.init_val = init_val if init_val != -1 else None

    def init_conv(self):
        if not self.init_val:
            return

        nn.init.uniform_(self.conv_h.weight, -self.init_val, self.init_val)
        nn.init.uniform_(self.conv_c.weight, -self.init_val, self.init_val)
        nn.init.uniform_(self.conv_m.weight, -self.init_val, self.init_val)
        nn.init.uniform_(self.conv_x.weight, -self.init_val, self.init_val)
        nn.init.uniform_(self.conv_c2.weight, -self.init_val, self.init_val)
        nn.init.uniform_(self.conv_o.weight, -self.init_val, self.init_val)
        nn.init.uniform_(self.conv_h2.weight, -self.init_val, self.init_val)

    def run_layer_norm(self, x, ln):
        idx = list(range(self.num_dims + 2))
        return ln(x.permute(0, *idx[2:], 1)).permute(0, -1, *idx[1:-1])

    def init_state(self, x, num_channels):
        dims = x.shape

        if len(dims) == self.num_dims + 2:
            b, rest = dims[0], dims[2:]
        else:
            raise ValueError('input tensor should be rank {}.'.format(
                self.num_dims + 2))

        return torch.zeros([b, num_channels, *rest],
                           dtype=x.dtype, device=x.device)

    def forward(self, x, h=None, c=None, m=None):
        if h is None:
            h = self.init_state(x, self.out_channels)
        if c is None:
            c = self.init_state(x, self.out_channels)
        if m is None:
            m = self.init_state(x, self.in_channels)

        h_cc = self.conv_h(h)
        c_cc = self.conv_c(c)
        m_cc = self.conv_m(m)

        if self.layer_norm:
            h_cc = self.run_layer_norm(h_cc, self.ln_h)
            c_cc = self.run_layer_norm(c_cc, self.ln_c)
            m_cc = self.run_layer_norm(m_cc, self.ln_m)

        i_h, g_h, f_h, o_h = torch.split(h_cc, self.out_channels, 1)
        i_c, g_c, f_c = torch.split(c_cc, self.out_channels, 1)
        i_m, f_m, m_m = torch.split(m_cc, self.out_channels, 1)

        if x is None:
            i = torch.sigmoid(i_h + i_c)
            f = torch.sigmoid(f_h + f_c + self.forget_bias)
            g = torch.tanh(g_h + g_c)
        else:
            x_cc = self.conv_x(x)
            if self.layer_norm:
                x_cc = self.run_layer_norm(x_cc, self.ln_x)

            i_x, g_x, f_x, o_x, i_x_, g_x_, f_x_ = \
                torch.split(x_cc, self.out_channels, 1)

            i = torch.sigmoid(i_x + i_h + i_c)
            f = torch.sigmoid(f_x + f_h + f_c + self.forget_bias)
            g = torch.tanh(g_x + g_h + g_c)

        c_new = f * c + i * g

        c2m = self.conv_c2(c_new)
        if self.layer_norm:
            c2m = self.run_layer_norm(c2m, self.ln_c2)

        i_c, g_c, f_c, o_c = torch.split(c2m, self.out_channels, 1)

        if x is None:
            ii = torch.sigmoid(i_c + i_m)
            ff = torch.sigmoid(f_c + f_m + self.forget_bias)
            gg = torch.tanh(g_c)
        else:
            ii = torch.sigmoid(i_c + i_x_ + i_m)
            ff = torch.sigmoid(f_c + f_x_ + f_m + self.forget_bias)
            gg = torch.tanh(g_c + g_x_)

        m_new = ff * torch.tanh(m_m) + ii * gg

        o_m = self.conv_o(m_new)
        if self.layer_norm:
            o_m = self.run_layer_norm(o_m, self.ln_o)

        if x is None:
            o = torch.tanh(o_h + o_c + o_m)
        else:
            o = torch.tanh(o_x + o_h + o_c + o_m)

        cell = torch.cat([c_new, m_new], 1)
        cell = self.conv_h2(cell)

        h_new = o * torch.tanh(cell)

        return h_new, c_new, m_new


class CausalLSTMCell2d(CausalLSTMCellBase):
    def __init__(self, filter_size, in_channels, out_channels,
                 forget_bias=1.0, layer_norm=False):
        super(CausalLSTMCell2d, self).__init__(
            filter_size, in_channels, out_channels,
            forget_bias, layer_norm)

        self.num_dims = 2

        self.conv_h = \
            nn.Conv2d(out_channels, out_channels * 4, filter_size,
                      stride=1, padding=1, padding_mode='replicate')
        self.conv_c = \
            nn.Conv2d(out_channels, out_channels * 3, filter_size,
                      stride=1, padding=1, padding_mode='replicate')
        self.conv_m = \
            nn.Conv2d(in_channels, out_channels * 3, filter_size,
                      stride=1, padding=1, padding_mode='replicate')
        self.conv_x = \
            nn.Conv2d(in_channels, out_channels * 7, filter_size,
                      stride=1, padding=1, padding_mode='replicate')
        self.conv_c2 = \
            nn.Conv2d(out_channels, out_channels * 4, filter_size,
                      stride=1, padding=1, padding_mode='replicate')
        self.conv_o = \
            nn.Conv2d(out_channels, out_channels, filter_size,
                      stride=1, padding=1, padding_mode='replicate')
        self.conv_h2 = \
            nn.Conv2d(out_channels * 2, out_channels, 1,
                      stride=1, padding=0, padding_mode='replicate')

        self.init_conv()


class CausalLSTMCell3d(CausalLSTMCellBase):
    def __init__(self, filter_size, in_channels, out_channels,
                 forget_bias=1.0, layer_norm=False):
        super(CausalLSTMCell3d, self).__init__(
            filter_size, in_channels, out_channels,
            forget_bias, layer_norm)

        self.num_dims = 3

        self.conv_h = \
            nn.Conv3d(out_channels, out_channels * 4, filter_size,
                      stride=1, padding=1, padding_mode='replicate')
        self.conv_c = \
            nn.Conv3d(out_channels, out_channels * 3, filter_size,
                      stride=1, padding=1, padding_mode='replicate')
        self.conv_m = \
            nn.Conv3d(in_channels, out_channels * 3, filter_size,
                      stride=1, padding=1, padding_mode='replicate')
        self.conv_x = \
            nn.Conv3d(in_channels, out_channels * 7, filter_size,
                      stride=1, padding=1, padding_mode='replicate')
        self.conv_c2 = \
            nn.Conv3d(out_channels, out_channels * 4, filter_size,
                      stride=1, padding=1, padding_mode='replicate')
        self.conv_o = \
            nn.Conv3d(out_channels, out_channels, filter_size,
                      stride=1, padding=1, padding_mode='replicate')
        self.conv_h2 = \
            nn.Conv3d(out_channels * 2, out_channels, 1,
                      stride=1, padding=0, padding_mode='replicate')

        self.init_conv()
