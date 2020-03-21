## Overview

This is the unofficial Pytorch implementation of paper *LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks*.

设$K$为要量化到的bit数。量化函数为：$Q(x, \mathbf{v})=\mathbf{v^T}\mathbf{e}_l$，其中$\mathbf{v}\in\mathbb{R}^K$，$l=1,...,2^K$，$\mathbf{e}_l\in\{-1,1\}^K$。也就是说用$K$个实数值来组成$x$。这样仍然能保证向量乘的过程中使用xonr-count，同时扩大了解空间。

只量化权重效果很好。同时量化权重和激活值效果不如论文里说的。不知道为啥。

## Usage

```bash
python train.py
```

There're several arguments to pass but there's no time to explain them.

resnet18的lr最开始设成0.01

## References

- https://github.com/microsoft/LQ-Nets (Official implementation with Tensorflow)