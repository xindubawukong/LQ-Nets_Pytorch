## Overview

This is the unofficial Pytorch implementation of paper *LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks*.

Only tested on cifar10 dataset with vgg11 model.

从cifar10上的结果来看的话应该问题不大。

## Usage

```bash
python train.py
```

There're several arguments to pass but there's no time to explain them.

resnet18的lr最开始设成0.01

## Results

All models are trained from scratch.

### cifar10 results

|Network|Weight Bits|Activation Bits|Top-1 Accuracy|Top-5 Accuracy|
|:---:|:---:|:---:|:---:|:---:|
|vgg11_bn|32|32|86.77|99.21|
|vgg11_bn|1|32|84.1|99.05|
|vgg11_bn|2|32|86.25|99.13|
|vgg11_bn|3|32|86.48|99.39|
|vgg11_bn|4|32|86.88|99.29|

## References

- https://github.com/microsoft/LQ-Nets (Official implementation with Tensorflow)