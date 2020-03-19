import torch
import torch.nn as nn
import numpy as np


NORM_PPF_0_75 = 0.6745


class WeightQuantizer(object):

    def __init__(self, nbit, num_filters):
        self.nbit = nbit
        if self.nbit == 0:
            return
        self.num_filters = num_filters
        init_basis = []
        n = num_filters * 3 * 3
        base = NORM_PPF_0_75 * ((2. / n) ** 0.5) / (2 ** (nbit - 1))
        for i in range(num_filters):
            t = [(2 ** j) * base for j in range(nbit)]
            t.reverse()  # test
            init_basis.append(t)
        self.basis = torch.Tensor(init_basis)
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
        self.level_multiplier = torch.Tensor(init_level_multiplier)
        # initialize threshold multiplier
        init_thrs_multiplier = []
        for i in range(1, num_levels):
            thrs_multiplier_i = [0. for j in range(num_levels)]
            thrs_multiplier_i[i - 1] = 0.5
            thrs_multiplier_i[i] = 0.5
            init_thrs_multiplier.append(thrs_multiplier_i)
        self.thrs_multiplier = torch.Tensor(init_thrs_multiplier)
        # print('level_multiplier:', self.level_multiplier)
        # print('thrs_multiplier:', self.thrs_multiplier)
    
    def quant(self, x, training=False):
        if self.nbit == 0:
            return x
        if x.is_cuda:
            self.basis = self.basis.cuda()
            self.level_multiplier = self.level_multiplier.cuda()
            self.thrs_multiplier = self.thrs_multiplier.cuda()
        nbit = self.nbit
        num_filters = self.num_filters
        num_levels = 2 ** self.nbit

        levels = torch.mm(self.basis, self.level_multiplier.t())
        levels, sort_id = torch.topk(levels, k=num_levels, dim=1, largest=False)
        # print('levels:', levels)
        thrs = torch.mm(levels, self.thrs_multiplier.t())

        reshape_x = x.view(num_filters, -1)

        level_codes_channelwise = torch.zeros(num_filters, num_levels, nbit)
        if x.is_cuda:
            level_codes_channelwise = level_codes_channelwise.cuda()
        for i in range(num_levels):
            eq = (sort_id == i).unsqueeze(2).expand(num_filters, num_levels, nbit)
            level_codes_channelwise = torch.where(eq, self.level_multiplier[i].view(-1).expand_as(level_codes_channelwise), level_codes_channelwise)
        # print(level_codes_channelwise.size(), level_codes_channelwise[0][0], level_codes_channelwise[0][1])
        y = torch.zeros_like(reshape_x) + levels[:, 0].view(-1, 1)
        bits_y = torch.zeros(list(reshape_x.size()) + [nbit]) - 1
        if x.is_cuda:
            bits_y = bits_y.cuda()
        for i in range(num_levels - 1):
            gt = reshape_x >= thrs[:, i].view(-1, 1)
            y = torch.where(gt, levels[:, i + 1].view(-1, 1).expand_as(y), y)
            tt = gt.unsqueeze(2).expand(list(reshape_x.size()) + [nbit])
            bits_y = torch.where(tt, level_codes_channelwise[:, i + 1].view(num_filters, 1, nbit).expand_as(bits_y), bits_y)
        if training:
            # bits_y: num_filters * in_channel * kernel_size * kernel_size * nbit
            BT = bits_y.view(num_filters, -1, nbit)
            B = BT.transpose(1, 2)
            BxBT = torch.bmm(B, BT)
            eps = BxBT.abs().max() * 0.0001 * torch.eye(nbit) * torch.randn(nbit, nbit)
            if x.is_cuda:
                eps = eps.cuda()
            BxBT += eps
            BxBT_inv = torch.inverse(BxBT)
            BxX = torch.bmm(B, x.view(num_filters, -1, 1))
            new_basis = torch.bmm(BxBT_inv, BxX)
            self.basis = new_basis.view(num_filters, nbit)
        y = y.view_as(x)
        return y


class ActivationQuantizer(object):

    def __init__(self, nbit):
        self.nbit = nbit
        if self.nbit == 0:
            return
        self.weight_quantizer = WeightQuantizer(nbit, num_filters=1)
    
    def quant(self, x, training=False):
        if self.nbit == 0:
            return x
        t = x.view(1, -1)
        y = self.weight_quantizer.quant(t, training)
        y = y.view_as(x)
        return x + x.detach() * -1 + y.detach()


class QuantConv2d(nn.Conv2d):

    def __init__(self, w_bit=0, a_bit=0, **kwargs):
        super().__init__(**kwargs)
        self.weight.org = self.weight.data.clone()
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.weight_quantizer = WeightQuantizer(w_bit, self.out_channels)
        self.activation_quantizer = ActivationQuantizer(a_bit)
    
    def forward(self, x):
        x = self.activation_quantizer.quant(x)
        self.weight.data = self.weight_quantizer.quant(self.weight.data, training=self.training)
        y = nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y


if __name__ == '__main__':
    torch.manual_seed(0)
    l = QuantConv2d(w_bit=2, a_bit=3, in_channels=3, out_channels=1, kernel_size=2)
    print(l)
    if hasattr(l.weight_quantizer, 'basis'):
        print(l.weight_quantizer.basis)
    print(l.weight)
    l.train()
    x = torch.randn(5, 3, 7, 7)
    x.requires_grad = True
    y = l(x)
    print(y.size())
    loss = y.mean()
    loss.backward()
    print(x.grad)