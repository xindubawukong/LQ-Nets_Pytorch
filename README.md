## Overview

This is the unofficial Pytorch implementation of paper *LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks*.

只量化权重效果很好。同时量化权重和激活值效果不如论文里说的。不知道为啥。

## Usage

```bash
python train.py
```

There're several arguments to pass but there's no time to explain them.

resnet18的lr最开始设成0.01

## References

- https://github.com/microsoft/LQ-Nets (Official implementation with Tensorflow)