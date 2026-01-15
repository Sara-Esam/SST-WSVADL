# SST-WSVADL: Sparse Spatio-Temporal Baseline and Patch-Level Benchmark for Weakly Supervised Video Anomaly Detection

Official implementation of the Sparse Spatio-Temporal Baseline and Patch-Level Benchmark for Weakly Supervised Video Anomaly Detection.

## Overview

This repository provides the code for a two-stage video anomaly detection
![Two-Stage VAD Architecture](./images/SST-WSVADL_framework.png)

**Note**: This code is based on the original [UR-DMU](https://github.com/henrryzh1/UR-DMU) and [STPrivacy](https://github.com/ming1993li/stprivacy) repositories. AI was used for refactoring this repository.

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA-capable GPU

### Setup

1. Clone or copy this repository:
```bash
cd SST-WSVADL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

### UCF-Crime Dataset

1. Download the UCF-Crime dataset
2. Extract video features using I3D or VideoMAEv2
3. Prepare the dataset list files in the `list/` directory
4. Update paths in the configuration and dataset

## Usage

### Training Structure

The training script structure demonstrates the two-stage training approach:
```bash
# Basic command structure (update paths and ensure dependencies)
python train.py \
    --dataset ucf \
    --batch_size 16 \
    --segment_length 16 \
    --num_tubelet 8 \
    --patch_size 16 \
    --resize 128 128 \
    --output_path ./outputs \
    --model_path ./models/exp_x \
    --video_root /path/to/videos \
    --test_file /path/to/ground_truth.npy
```

### Key Parameters

- `--dataset`: Dataset name (default: ucf)
- `--batch_size`: Batch size for training (default: 16)
- `--segment_length`: Number of frames per segment (default: 16)
- `--num_tubelet`: Number of temporal tubelets (default: 8)
- `--patch_size`: Patch size for spatial division (default: 16)
- `--resize`: Resize dimensions for frames (default: 128 128)
- `--lr`: Learning rate (default: 0.0001)
- `--num_epochs`: Number of training epochs (default: 4000)
- `--token_ratio`: Token pruning ratios during test (default: 1.0 1.0 1.0)
- `--cross_attention`: Enable patch-to-snippet attention
- `--pretrained_point`: Use pretrained UR-DMU model for initializing the weights

## Acknowledgments

This implementation is based on:
- UR-DMU: Dual Memory Units with Uncertainty Regulation
- STPrivacy: Spatio-Temporal Privacy-preserving Action Recognition

