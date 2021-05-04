import math

import torch
import torch.nn as nn


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, kernel_size, in_shape, bias=True):
        """
        """
        super(SpatioTemporalLSTMCell, self).__init__()

        self.in_channel = in_channel
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        self.batch = in_shape[0]
        self.channel = in_shape[1]
        self.height = in_shape[2]
        self.width = in_shape[3]
        self.padding = kernel_size // 2
        self.bias = bias

        self.t_cc = nn.Conv2d(num_hidden, self.num_hidden * 4,
                              kernel_size=self.kernel_size,
                              stride=1, padding=self.padding,
                              bias=bias
                             )

        self.s_cc = nn.Conv2d(num_hidden, self.num_hidden * 4,
                              kernel_size=self.kernel_size,
                              stride=1, padding=self.padding,
                              bias=bias
                             )

        self.x_cc = nn.Conv2d(in_channel, self.num_hidden * 4,
                              kernel_size=self.kernel_size,
                              stride=1, padding=self.padding,
                              bias=bias
                              )

        self.last = nn.Conv2d(num_hidden*2, self.num_hidden,
                              kernel_size=1, stride=1, padding=0,
                              bias=bias
                              )


    def init_state(self):
        return torch.zeros([self.batch, self.num_hidden, self.height, self.width],
                        dtype=torch.float32)


    def forward(self, x, h, c, m):
        if h is None:
            h = self.init_state().to(x.device)
        if c is None:
            c = self.init_state().to(x.device)
        if m is None:
            m = self.init_state().to(x.device)

        t_cc = self.t_cc(h)
        s_cc = self.s_cc(m)
        x_cc = self.x_cc(x)

        i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, dim=1)
        i_t, g_t, f_t, o_t = torch.split(t_cc, self.num_hidden, dim=1)
        i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, dim=1)

        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f = torch.sigmoid(f_x + f_t)
        f_ = torch.sigmoid(f_x + f_s)
        o = torch.sigmoid(o_x + o_t + o_s)
        new_m = f_ * m + i_ * g_
        new_c = f * c + i * g

        cell = torch.cat([new_m, new_c], dim=1)
        cell = self.last(cell)

        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m


class MIMS(nn.Module):
    def __init__(self, in_channel, num_hidden, kernel_size, in_shape, stride=1, bias=True):
        super(MIMS, self).__init__()

        self.in_channel = in_channel
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.batch = in_shape[0]
        self.channel = in_shape[1]
        self.height = in_shape[2]
        self.width = in_shape[3]
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv_h = nn.Sequential(nn.Conv2d(self.num_hidden, self.num_hidden * 4,
                                              kernel_size=self.kernel_size,
                                              stride=self.stride, padding=self.padding,
                                              bias=self.bias
                                              ),
                                   #nn.LayerNorm([num_hidden*4, self.height, self.width])
                                  )

        self.conv_x = nn.Sequential(nn.Conv2d(self.in_channel, self.num_hidden * 4,
                                            kernel_size=self.kernel_size,
                                            stride=self.stride, padding=self.padding,
                                            bias=self.bias
                                            ),
                                   #nn.LayerNorm([num_hidden*4, self.height, self.width])
                                  )

        ct_weight_ = torch.empty((self.num_hidden*2, self.height, self.width))
        self.ct_weight = nn.Parameter(nn.init.kaiming_normal_(ct_weight_))
        oc_weight_ = torch.empty((self.num_hidden, self.height, self.width))
        self.oc_weight = nn.Parameter(nn.init.kaiming_normal_(oc_weight_))

    def init_state(self):
        return torch.zeros([self.batch, self.num_hidden, self.height, self.width],
                            dtype=torch.float32)

    def forward(self, x, h_t, c_t):
        if h_t is None:
            h_t = self.init_state().to(x.device)
        if c_t is None:
            c_t = self.init_state().to(x.device)

        h_concat = self.conv_h(h_t)
        g_h, i_h, f_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        ct_activation = c_t.repeat([1,2,1,1]) * self.ct_weight
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)
        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x != None:
            x_concat = self.conv_x(x)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)

            i_ += i_x
            f_ += f_x
            g_ += g_x
            o_ += o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = o_ + c_new * self.oc_weight

        h_new = torch.sigmoid(o_c) * torch.tanh(c_new)

        return h_new, c_new


class MIMBlock(nn.Module):
    def __init__(self, in_channel, num_hidden, kernel_size, in_shape,
                 stride=1, bias=True):
        super(MIMBlock, self).__init__()
        self.in_channel = in_channel
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.batch = in_shape[0]
        self.height = in_shape[2]
        self.width = in_shape[3]
        self.padding = kernel_size // 2
        self.bias = bias

        self.t_cc = nn.Sequential(nn.Conv2d(num_hidden, num_hidden*3,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride, padding=self.padding,
                                          bias=self.bias
                                          ),
                                #nn.LayerNorm([num_hidden*3, self.height, self.width])
                                )

        self.s_cc = nn.Sequential(nn.Conv2d(num_hidden, num_hidden*4,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride, padding=self.padding,
                                          bias=self.bias
                                          ),
                                 #nn.LayerNorm([num_hidden*4, self.height, self.width])
                                )

        self.x_cc = nn.Sequential(nn.Conv2d(in_channel, num_hidden*4,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride, padding=self.padding,
                                          bias=self.bias
                                          ),
                                 #nn.LayerNorm([num_hidden*4, self.height, self.width])
                                 )

        self.mims = MIMS(in_channel, num_hidden, kernel_size, in_shape,
                         stride, bias)
        self.last = nn.Conv2d(num_hidden*2, num_hidden, 1, 1, padding=0)

    def init_state(self):
        return torch.zeros([self.batch, self.num_hidden, self.height, self.width],
                        dtype=torch.float32)

    def forward(self, x, diff_h, h, c, m, convlstm_c):
        if h is None:
            h = self.init_state().to(x.device)
        if c is None:
            c = self.init_state().to(x.device)
        if m is None:
            m = self.init_state().to(x.device)
        if convlstm_c is None:
            convlstm_c = self.init_state().to(x.device)
        if diff_h is None:
            diff_h = torch.zeros_like(h, device=x.device)

        t_cc = self.t_cc(h)
        s_cc = self.s_cc(m)
        x_cc = self.x_cc(x)

        i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, dim=1)
        i_t, g_t, o_t = torch.split(t_cc, self.num_hidden, dim=1)
        i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, dim=1)

        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f_ = torch.sigmoid(f_x + f_s)
        o = torch.sigmoid(o_x + o_t + o_s)
        new_m = f_ * m + i_ * g_

        c, convlstm_c = self.mims(diff_h, c, convlstm_c)

        new_c = c + i * g
        cell = torch.cat([new_c, new_m], dim=1)
        cell = self.last(cell)
        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m, convlstm_c


class MIMN(nn.Module):
    def __init__(self, in_channel, num_hidden, kernel_size, in_shape,
                 stride=1, bias=True):
        """
        """
        super(MIMN, self).__init__()
        self.in_channel = in_channel
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.batch = in_shape[0]
        self.height = in_shape[2]
        self.width = in_shape[3]
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv_h = nn.Sequential(
                            nn.Conv2d(num_hidden, self.num_hidden * 4,
                                kernel_size=self.kernel_size,
                                stride=1, padding=self.padding,
                                bias=self.bias),
                            #nn.LayerNorm([num_hidden*4, self.height, self.width])
                            )

        self.conv_x = nn.Sequential(
                            nn.Conv2d(in_channel, self.num_hidden * 4,
                                kernel_size=self.kernel_size,
                                stride=1, padding=self.padding,
                                bias=self.bias
                                ),
                            #nn.LayerNorm([num_hidden*4, self.height, self.width])
                            )

        ct_weight_ = torch.empty((self.num_hidden*2, self.height, self.width))
        self.ct_weight = nn.Parameter(nn.init.kaiming_normal_(ct_weight_))

        oc_weight_ = torch.empty((self.num_hidden, self.height, self.width))
        self.oc_weight = nn.init.kaiming_normal_(oc_weight_)


    def init_state(self):
        shape = [self.batch, self.num_hidden, self.height, self.width]
        return torch.zeros(shape, dtype=torch.float32)

    def forward(self, x, h_t, c_t):
        if h_t is None:
            h_t = self.init_state().to(x.device)
        if c_t is None:
            c_t = self.init_state().to(x.device)

        h_concat = self.conv_h(h_t)

        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        ct_activation = c_t.repeat([1,2,1,1]) * self.ct_weight
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x != None:
            x_concat = self.conv_x(x)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)

            i_ += i_x
            f_ += f_x
            g_ += g_x
            o_ += o_x

        g_ = torch.tanh(g_)
        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_)
        c_new = f_ * c_t + i_ * g_

        o_c = torch.sigmoid(o_ + c_new * self.oc_weight.to(x.device))

        h_new = o_c * torch.tanh(c_new)

        return h_new, c_new


class MIM(nn.Module):
    def __init__(self, input_dims, out_dims, in_shape, hidden_dim=[32, 32],
                 kernel_size=3, stride=1, total_length=10, input_length=5,
                 lr=0.001):
        """
        :param input_dims(int): input channel
        :param out_dims(int): output channel
        :param in_shape(list): input shape
        :param hidden_dim(list): hidden layers channel
        :param kernel_size(int): kernel size
        :param stride(int): stride
        :param total_length(int): total number of frames
        :param input_length(int): number of input frames
        :param lr(float): learning rate
        """
        super(MIM, self).__init__()

        self.learning_rate = lr
        self.input_dims = input_dims
        self.out_dims = out_dims
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_layers = len(hidden_dim)
        self.total_length = total_length
        self.input_length = input_length
        self.in_shape = in_shape
        self.batch = in_shape[0]
        self.height = in_shape[2]
        self.width = in_shape[3]

        # stationarity
        self.stlstm_layer = nn.ModuleList([])
        for i in range(self.num_layers):
            if i < 1:
                self.stlstm_layer.append(SpatioTemporalLSTMCell(self.input_dims, self.hidden_dim[i],
                                                                kernel_size=kernel_size,
                                                                in_shape=in_shape
                                                               )
                                        )
            else:
                self.stlstm_layer.append(MIMBlock(self.hidden_dim[i-1], self.hidden_dim[i],
                                                  kernel_size=kernel_size, stride=stride,
                                                  in_shape=in_shape
                                                 )
                                        )

        # non-stationarity
        self.stlstm_layer_diff = nn.ModuleList([])
        for i in range(self.num_layers-1):
            self.stlstm_layer_diff.append(MIMN(self.hidden_dim[i], self.hidden_dim[i+1],
                                               kernel_size=kernel_size, stride=stride,
                                               in_shape=in_shape
                                              )
                                         )

        self.last = nn.Conv2d(self.hidden_dim[-1], self.out_dims, 1, 1, 0)

    def forward(self, frames, ss_bool=None):
        st_memory = None
        cell_state = [None] * self.num_layers
        hidden_state = [None] * self.num_layers
        cell_state_diff = [None] * (self.num_layers - 1)
        hidden_state_diff = [None] * (self.num_layers - 1)
        convlstm_c = [None] * (self.num_layers - 1)

        gen_imgs = []
        for ts in range(self.total_length - 1):
            if ts < self.input_length:
                x_gen = frames[:, ts]
            else:
                x_gen = ss_bool[:, ts-self.input_length] * frames[:, ts] + (1 - ss_bool[:, ts-self.input_length]) * x_gen

            preh = hidden_state[0]
            # 1st layer(convlstm)
            hidden_state[0], cell_state[0], st_memory = self.stlstm_layer[0](
                    x_gen, hidden_state[0], cell_state[0], st_memory)

            # higher layer(mim)
            for i in range(1, self.num_layers):
                if ts > 0:
                    if i == 1:
                        hidden_state_diff[i-1], cell_state_diff[i-1] = self.stlstm_layer_diff[i-1](hidden_state[i-1] - preh,
                                                                                                   hidden_state_diff[i-1],
                                                                                                   cell_state_diff[i-1])
                    else:
                        hidden_state_diff[i-1], cell_state_diff[i-1] = self.stlstm_layer_diff[i-1](
                            hidden_state_diff[i-2], hidden_state_diff[i-1], cell_state_diff[i-1])
                else:
                    self.stlstm_layer_diff[i-1](torch.zeros_like(hidden_state[i-1]), None, None)

                preh = hidden_state[i]
                hidden_state[i], cell_state[i], st_memory, convlstm_c[i-1] = self.stlstm_layer[i](
                    hidden_state[i-1], hidden_state_diff[i-1], hidden_state[i], cell_state[i], st_memory, convlstm_c[i-1])

            x_gen = self.last(hidden_state[-1])
            gen_imgs.append(x_gen)

        return torch.stack(gen_imgs, dim=1)

