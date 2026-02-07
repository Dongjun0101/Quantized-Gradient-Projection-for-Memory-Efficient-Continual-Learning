# [ICLR 2026] Quantized Gradient Projection for Memory-Efficient Continual Learning
This repository contains the implementation for the paper [Quantized Gradient Projection for Memory-Efficient Continual Learning, The Fourteenth International Conference on Learning Representations (ICLR 2026)](https://openreview.net/forum?id=xJtxpJ6QdD).


## TL;DR
### We propose QGPM, a memory-efficient and privacy-preserving continual learning framework that compresses task subspaces via quantization.

## Requirements
```
pip install -r requirements.txt
```

## 10-split CIFAR100
```
python qgpm_alexnet.py
```

## 5-Datasets
```
python qgpm_resnet.py
```

## 10-split miniImageNet
```
python qgpm_vit.py
```

## Acknowledgment
Parts of this codebase were adapted from [GPM](https://github.com/sahagobinda/GPM/tree/main) 

