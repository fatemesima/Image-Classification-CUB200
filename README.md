# EfficientNet on CUB-200-2011

This repository trains **EfficientNet-B0** on the [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) bird dataset.

## Features

- Stage 1: Only train the fully-connected layer (`_fc`).
- Stage 2: Unfreeze the entire EfficientNet backbone and continue training.
- Logging with `logging` module.
- TensorBoard support.

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- torchvision
- efficientnet_pytorch
- TensorBoard

## Usage

```bash
python train_efficientnet.py
