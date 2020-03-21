import torch
import torch.nn as nn
import numpy as np


NORM_PPF_0_75 = 0.6745


class WeightQuantizer(nn.Module):

    def __init__(self, nbit, num_filters, method='QEM'):
        super().__init__()
        self.nbit = nbit
        if self.nbit == 0:
            return
        self.num_filters = num_filters
        self.method = method
        init_basis = []
        n = num_filters * 3 * 3
        base = NORM_PPF_0_75 * ((2. / n) ** 0.5) / (2 ** (nbit - 1))
        for i in range(num_filters):
            t = [(2 ** j) * base for j in range(nbit)]
            init_basis.append(t)
        if method == 'QEM':
            self.basis = nn.Parameter(torch.Tensor(init_basis), requires_grad=False)
        else:
            self.basis = nn.Parameter(torch.Tensor(init_basis), requires_grad=True)
        # print('basis:', self.basis)
        num_levels = 2 ** nbit
        # initialize level multiplier
        init_level_multiplier = []
        for i in range(num_levels):
            level_multiplier_i = [0. for j in range(nbit)]
            level_number = i
            for j in range(nbit):
                binary_code = level_number % 2
                if binary_code == 0:
                    binary_code = -1
                level_multiplier_i[j] = float(binary_code)
                level_number = level_number // 2
            init_level_multiplier.append(level_multiplier_i)
        self.level_multiplier = nn.Parameter(torch.Tensor(init_level_multiplier), requires_grad=False)
        # initialize threshold multiplier
        init_thrs_multiplier = []
        for i in range(1, num_levels):
            thrs_multiplier_i = [0. for j in range(num_levels)]
            thrs_multiplier_i[i - 1] = 0.5
            thrs_multiplier_i[i] = 0.5
            init_thrs_multiplier.append(thrs_multiplier_i)
        self.thrs_multiplier = nn.Parameter(torch.Tensor(init_thrs_multiplier), requires_grad=False)
        # print('level_multiplier:', self.level_multiplier)
        # print('thrs_multiplier:', self.thrs_multiplier)
        self.level_codes_channelwise = nn.Parameter(torch.zeros(num_filters, num_levels, nbit), requires_grad=False)
        self.eps = nn.Parameter(torch.eye(nbit) * 0.01, requires_grad=False)
        self.record = []
    
    def forward(self, x, training=False):
        if self.nbit == 0:
            return x
        nbit = self.nbit
        num_filters = self.num_filters
        num_levels = 2 ** self.nbit

        assert x.size(0) == num_filters

        levels = torch.mm(self.basis, self.level_multiplier.t())
        levels, sort_id = torch.topk(levels, k=num_levels, dim=1, largest=False)
        # print('levels:', levels)
        thrs = torch.mm(levels, self.thrs_multiplier.t())

        reshape_x = x.view(num_filters, -1)

        level_codes_channelwise = self.level_codes_channelwise
        for i in range(num_levels):
            eq = (sort_id == i).unsqueeze(2).expand(num_filters, num_levels, nbit)
            level_codes_channelwise = torch.where(eq, self.level_multiplier[i].view(-1).expand_as(level_codes_channelwise), level_codes_channelwise)
        # print(level_codes_channelwise.size(), level_codes_channelwise[0][0], level_codes_channelwise[0][1])
        y = torch.zeros_like(reshape_x) + levels[:, 0].view(-1, 1)
        bits_y = reshape_x.clone().unsqueeze(2).expand(num_filters, reshape_x.size(1), nbit)
        bits_y = bits_y * 0 - 1
        for i in range(num_levels - 1):
            gt = reshape_x >= thrs[:, i].view(-1, 1)
            y = torch.where(gt, levels[:, i + 1].view(-1, 1).expand_as(y), y)
            tt = gt.unsqueeze(2).expand(list(reshape_x.size()) + [nbit])
            bits_y = torch.where(tt, level_codes_channelwise[:, i + 1].view(num_filters, 1, nbit).expand_as(bits_y), bits_y)
        if training and self.method == 'QEM':
            # bits_y: num_filters * in_channel * kernel_size * kernel_size * nbit
            BT = bits_y.view(num_filters, -1, nbit)
            B = BT.transpose(1, 2)
            BxBT = torch.bmm(B, BT)
            BxBT += self.eps
            BxBT_inv = torch.inverse(BxBT)
            BxX = torch.bmm(B, x.view(num_filters, -1, 1))
            new_basis = torch.bmm(BxBT_inv, BxX)
            new_basis = torch.topk(new_basis, k=nbit, dim=1, largest=False)[0]
            self.record.append(new_basis.view(num_filters, nbit).unsqueeze(0))
        y = y.view_as(x)
        return y.detach() + x + x.detach() * -1, [levels.min().item(), levels.max().item()]


class ActivationQuantizer(nn.Module):

    def __init__(self, nbit, method='QEM'):
        super().__init__()
        self.nbit = nbit
        if self.nbit == 0:
            return
        self.weight_quantizer = WeightQuantizer(nbit, num_filters=1, method=method)
    
    def forward(self, x, training=False):
        if self.nbit == 0:
            return x
        t = x.view(1, -1)
        y, l = self.weight_quantizer(t, training)
        y = y.view_as(x)
        t = torch.clamp(x, l[0], l[1])
        return y.detach() + t + t.detach() * -1


class QuantConv2d(nn.Conv2d):

    def __init__(self, w_bit=0, a_bit=0, method='QEM', **kwargs):
        super().__init__(**kwargs)
        self.weight.org = self.weight.data.clone()
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.weight_quantizer = WeightQuantizer(w_bit, self.out_channels, method=method)
        self.activation_quantizer = ActivationQuantizer(a_bit, method=method)
    
    def forward(self, x):
        if (self.in_channels > 3):
            x = self.activation_quantizer(x, training=self.training)
            self.weight.data, _ = self.weight_quantizer(self.weight.data, training=self.training)
        y = nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y


if __name__ == '__main__':
    torch.manual_seed(0)
    # l = QuantConv2d(w_bit=2, a_bit=3, in_channels=3, out_channels=1, kernel_size=2)
    # print(l)
    # if hasattr(l.weight_quantizer, 'basis'):
    #     print(l.weight_quantizer.basis)
    # print(l.weight)
    # l.train()
    # x = torch.randn(5, 3, 7, 7)
    # x.requires_grad = True
    # y = l(x)
    # print(y.size())
    # loss = y.mean()
    # loss.backward()
    # print(x.grad)

    aa = ActivationQuantizer(3)

    x = torch.randn(5, 5)
    x.requires_grad = True
    print(x)
    y = aa.quant(x, True)
    print(y)
    y.backward(torch.ones(y.size()))
    print(x.grad)
    
    for i in range(10):
        y = aa.quant(x, True)
    y = aa.quant(x, True)
    print(y)
    y.backward(y)
    print(x.grad)