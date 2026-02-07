# [ICLR 2026] Quantized Gradient Projection for Memory-Efficient Continual Learning
By [Dongjun Kim](https://github.com/Dongjun0101), Seohyeon Cha, Huancheng Chen, Chianing Wang, and Haris Vikalo.

This repository contains the implementation for the paper [Quantized Gradient Projection for Memory-Efficient Continual Learning, The Fourteenth International Conference on Learning Representations (ICLR 2026)](https://openreview.net/forum?id=xJtxpJ6QdD).


## TL;DR
### We propose QGPM, a memory-efficient and privacy-preserving continual learning framework that compresses task subspaces via quantization.

## 10-split CIFAR100
```
python qgpm_alexnet.py
```

## 5-Datasets
```
python qgpm_resnet.py
```

## 10/20-split miniImageNet
* 10-split
```
python qgpm_vit.py --dataset=miniimagenet --num_tasks=10
```
* 20-split
```
python qgpm_vit.py --dataset=miniimagenet --num_tasks=20
```

## Acknowledgment
Parts of this codebase were adapted from [GPM](https://github.com/sahagobinda/GPM/tree/main) 

