## Overview

This is the unofficial Pytorch implementation of paper *LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks*.

Only weight quantization is implemented. Only conv layers are implemented because fc layers can be conveted to conv layers. Only tested on cifar10 dataset with vgg11 model.

## Usage

```bash
python train.py
```

There're several arguments to pass but there's no time to explain them.

## Results

### cifar10 results

|Network|Weight Bits|Activation Bits|Top-1 Accuracy|Top-5 Accuracy|
|:---:|:---:|:---:|:---:|:---:|
|vgg11_bn|32|32|86.77|99.21|
|vgg11_bn|3|32|84.72|99.17|

## References

- https://github.com/microsoft/LQ-Nets (Official implementation with Tensorflow)