# Basic Transformer

This repository is a minimal translation transformer written in PyTorch.
It is based on the original [hkproj/pytorch-transformer](https://github.com/hkproj/pytorch-transformer) project, 
with additional inline comments reflecting personal understanding.

## Repository structure

| File | Description |
|------|-------------|
| `config.py` | Provides training hyper-parameters and utilities for locating weight files. |
| `model.py` | Implementation of the Transformer architecture with detailed comments. |
| `train.py` | Training loop using the configuration and model; loads translation datasets and performs tokenization. |

The `train.py` script expects a `dataset` module providing a `BilingualDataset` class and a `causal_mask` helper.
These components are not included in this snapshot but mirror the original project.

## Quick start

1. Install requirements such as `torch`, `torchtext`, `tokenizers`, and `datasets`.
2. Adjust settings in `config.py` if necessary.
3. Run `python train.py` to begin training (requires GPU for practical performance).

---
This simple clone serves as a learning reference for building transformer models from scratch.
