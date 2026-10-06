# HRLR-Mixed-SimCLR: HR-LR Discrimination via Self-Supervised Learning

This project implements a self-supervised learning approach to distinguish high-resolution (HR) images from low-resolution (LR) or degraded images. The approach uses SimCLR (Simple Framework for Contrastive Learning of Visual Representations) for pretraining, followed by fine-tuning and linear evaluation to assess the quality of learned representations.

## Overview

The project consists of several components:

1. **Pretraining** (`run.py`): Self-supervised pretraining using SimCLR on HR image patches
2. **Fine-tuning** (`finetune_realsr.py`): Fine-tuning the pretrained encoder for HR/LR classification
3. **Linear Evaluation** (`linear_eval_realsr.py`): Linear probing to evaluate representation quality
4. **Utilities**: Helper functions for models, losses, optimizers, and data loading
5. **Additional Tools**: 
   - `tsne_viz.py`: t-SNE visualization of learned features

## Directory Structure

```
.
├── README.md
├── run.py                  # SimCLR pretraining script
├── finetune_realsr.py      # Fine-tuning script for HR/LR classification
├── linear_eval_realsr.py   # Linear evaluation script
├── data.py                 # Data loading for pretraining
├── resnet.py               # ResNet backbone with projection head
├── objective.py            # NT-Xent loss implementation
├── model_util.py           # Utility functions (LARS optimizer, projection head, etc.)
├── add_div2k_val.py        # Script to add validation DIV2K dataset
├── tsne_viz.py             # t-SNE visualization of learned features
├── checkpoints/            # Directory for pretraining checkpoints (created during training)
├── checkpoints_mixed/      # Directory for fine-tuning checkpoints
├── tsne_hr_lr_comparison.png # t-SNE visualization results
└── Dataset/
    ├── mixed_dataset/      # Training data (HR and LR images)
    └── val_dataset/        # Validation data (HR and LR image pairs)
```

## Dataset

The dataset consists of paired high-resolution (HR) and low-resolution (LR) images from multiple sources:
- Canon (RealSR dataset)
- DIV2K 
- Nikon (RealSR dataset)

Images are organized in two main directories:
- `Dataset/mixed_dataset/`: Training data containing both HR and LR images
- `Dataset/val_dataset/`: Validation data with paired HR/LR images

## Usage

### 1. Pretraining (SimCLR)

To pretrain the encoder using SimCLR on HR image patches:

```bash
python run.py \
  --hr_dir Dataset/mixed_dataset/all_images \
  --model_dir ./checkpoints \
  --patch_size 96 \
  --patches_per_image 50 \
  --resnet_depth 18 \
  --train_batch_size 256 \
  --train_epochs 100 \
  --learning_rate 0.3
```

This will save checkpoints to the specified model directory.

### 2. Fine-tuning

To fine-tune the pretrained encoder for HR/LR classification:

```bash
python finetune_realsr.py
```

The script will:
- Load a pretrained checkpoint (default: `./checkpoints_mixed/checkpoint_epoch0100.pth`)
- Fine-tune the encoder unfrozen with a small learning rate
- Train a linear classifier on top
- Save the best model based on validation accuracy

### 3. Linear Evaluation

To evaluate the quality of learned representations using linear probing:

```bash
python linear_eval_realsr.py
```

This script will:
- Extract features from the frozen pretrained encoder
- Train a logistic regression classifier on top
- Report classification accuracy and per-source breakdown

#### Additional Tools


#### t-SNE Visualization

To visualize the learned feature space using t-SNE:

```bash
python tsne_viz.py
```

This generates a visualization comparing the feature space of randomly initialized vs. pretrained encoders, saved as `tsne_hr_lr_comparison.png`.

## Key Features

- **SimCLR Implementation**: Complete implementation of the SimCLR framework for self-supervised learning
- **Custom Data Loader**: Efficiently loads and patches images on-the-fly without storing all patches in memory
- **LARS Optimizer**: Implements the Layer-wise Adaptive Rate Scaling optimizer used in SimCLR
- **Multiple Evaluation Approaches**: Both fine-tuning and linear evaluation to assess representation quality
- **Multi-source Support**: Handles images from different sources (Canon, DIV2K, Nikon) with appropriate naming conventions

## Technical Details

### Model Architecture
- **Backbone**: ResNet-18 (configurable to other depths)
- **Projection Head**: 3-layer MLP as used in SimCLR
- **Output**: 128-dimensional projection vectors

### Training Procedure
1. **Pretraining**: Self-supervised contrastive learning using NT-Xent loss
2. **Fine-tuning**: Supervised learning with unfrozen encoder and small learning rate
3. **Linear Evaluation**: Frozen encoder with linear classifier on top

### Loss Function
- **NT-Xent Loss**: Normalized temperature-scaled cross entropy loss
- **Temperature**: 0.5 (configurable)

### Optimizer
- **LARS**: Layer-wise Adaptive Rate Scaling optimizer with weight decay
- **Learning Rate Schedule**: Warmup followed by cosine annealing

## Files Description

### Core Scripts
- `run.py`: Main pretraining script implementing SimCLR
- `finetune_realsr.py`: Fine-tuning script for HR/LR classification task
- `linear_eval_realsr.py`: Linear evaluation script for assessing representation quality

### Modules
- `data.py`: Data loading and augmentation for pretraining
- `resnet.py`: ResNet backbone implementation with projection head
- `objective.py`: NT-Xent loss function implementation
- `model_util.py`: Projection head, LARS optimizer, and learning rate scheduling

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- scikit-learn (for linear evaluation and t-SNE)
- tqdm
- Pillow
- matplotlib (for visualization)

## Results

### SimCLR Pretraining Results
From the training logs (simclr_mixed_log.txt):
- **Model**: ResNet-18 + 3-layer projection head
- **Patch size**: 96x96
- **Batch size**: 256 (effective batch size with gradient accumulation)
- **Epochs**: 100
- **Optimizer**: LARS (official google-research formula)
- **Loss**: NT-Xent (sum, official formula)
- **Warmup**: 10 epochs + cosine decay
- **Augmentation**: crop, flip, color jitter, grayscale (NO blur)
- **Final Contrast Accuracy**: 0.7146 (71.46%)

### Fine-tuning Results
From the fine-tuning logs (finetune_realsr_log5.txt):
- **Best Validation Accuracy**: 90.53%
- **Per-source accuracy**:
  - Canon: 99.37% (2981/3000)
  - DIV2K: 74.90% (2247/3000)
  - Nikon: 97.30% (2919/3000)

### Linear Evaluation Results
From the linear evaluation logs (linear_eval_log2.txt):
- **Random encoder accuracy**: 80.27%
- **Pretrained encoder (val)**: 98.00%
- **Improvement over random**: +17.73%
- **Per-source accuracy**:
  - Canon: 99.53%
  - DIV2K: 95.54%
  - Nikon: 98.92%

### Comparison with Official SimCLR Benchmark
While the original SimCLR paper (Chen et al., 2020) focused on 1000-class ImageNet classification using a larger ResNet-50 backbone (achieving ~76.5% top-1 accuracy under linear evaluation), our project adapts the SimCLR framework for a binary HR vs LR patch classification task using a lighter ResNet-18 backbone. 

Our linear evaluation accuracy of **98.00%** demonstrates that the contrastive self-supervised objective is highly effective at capturing low-level image degradation features, providing a massive **+17.73%** gain over a random initialization baseline. Even with a much smaller architecture (ResNet-18 vs ResNet-50) and fewer training epochs, the learned representation quality validates the core findings of the original SimCLR paper, demonstrating that SimCLR is highly effective in a specialized image restoration context.

The pretrained representations effectively capture features that distinguish high-resolution (HR) from low-resolution (LR/degraded) images, achieving strong performance across different camera sources (Canon, DIV2K, Nikon).

## Acknowledgments

This implementation is based on the SimCLR framework from:
- Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
- "A Simple Framework for Contrastive Learning of Visual Representations"
- https://arxiv.org/abs/2002.05709