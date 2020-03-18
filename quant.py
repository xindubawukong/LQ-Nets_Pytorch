import torch
import torch.nn as nn


class MyConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
    
    def forward(self, x):
        return x


if __name__ == '__main__':
    net = MyConv2d(3,3,3)
    print(net)
    x = torch.randn(3, 5)
    print(x)
    print(net(x))