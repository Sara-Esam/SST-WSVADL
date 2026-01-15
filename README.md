# SST-WSVADL: Spatio-Temporal Weakly Supervised Video Anomaly Detection with Learning

A clean, refactored implementation of a two-stage weakly supervised video anomaly detection framework combining UR-DMU (Uncertainty-Regulated Dual Memory Units) with STPrivacy (Spatio-Temporal Privacy-preserving) for patch-level anomaly detection.

## Overview

This repository provides a **simplified and refactored** structure for a two-stage video anomaly detection model:
- **Stage 1**: Temporal-level anomaly detection using UR-DMU
- **Stage 2**: Patch-level anomaly detection using STPVAD (Spatio-Temporal Privacy VAD)

**Note**: This is a refactored version with clean structure and simplified code. For the full implementation, refer to the original UR-DMU repository. This version provides a clean structure that can be extended to a fully standalone implementation.

## Features

- Clean project structure and organization
- Simplified configuration management
- Two-stage training pipeline structure
- Patch-level spatio-temporal feature extraction (structure)
- Memory-based anomaly detection mechanism (structure)

## Current Status

This refactored version provides:
- ✅ Clean directory structure
- ✅ Simplified configuration system
- ✅ Simplified training script (structure)
- ✅ Documentation
- ⚠️ References original codebase for models and utilities

For full standalone implementation, see `README_REFACTORING.md` for a checklist of components to copy/refactor.

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA-capable GPU (recommended)
- Access to original UR-DMU and STPrivacy codebases

### Setup

1. Clone or copy this repository:
```bash
cd SST-WSVADL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure original codebases are accessible:
```bash
# The original UR-DMU and STPrivacy codebases should be accessible
# Update paths in train.py accordingly
```

## Dataset Preparation

### UCF-Crime Dataset

1. Download the UCF-Crime dataset
2. Extract video features using I3D or VideoMAEv2
3. Prepare the dataset list files in the `data/` directory
4. Update paths in the configuration

## Usage

**Note**: The current implementation provides a structure reference. For full functionality, complete the refactoring as outlined in `README_REFACTORING.md`.

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
    --model_path ./models \
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
- `--token_ratio`: Token pruning ratios (default: 1.0 1.0 1.0)
- `--cross_attention`: Enable cross-attention between stages
- `--pretrained_point`: Use pretrained UR-DMU model

## Project Structure

```
SST-WSVADL/
├── train.py              # Main training script
├── config.py             # Configuration management
├── models/               # Model definitions
│   ├── urdmu.py         # UR-DMU model
│   └── stpvad.py        # STPVAD model
├── data/                 # Dataset loaders
│   └── datasets.py      # Dataset classes
├── losses/               # Loss functions
│   └── vad_loss.py      # VAD loss definitions
├── utils/                # Utility functions
│   └── utils.py         # Helper functions
└── eval/                 # Evaluation scripts
    └── test.py          # Testing and evaluation
```

## Model Architecture

### Stage 1: UR-DMU (Temporal-level)
- Memory-based anomaly detection at temporal snippet level
- Dual memory units (normal and abnormal)
- Uncertainty regulation mechanism

### Stage 2: STPVAD (Patch-level)
- Spatio-temporal privacy-preserving feature extraction
- Patch-level anomaly scoring
- Cross-attention with temporal snippets

## Citation

If you use this code, please cite the original papers:

```bibtex
@inproceedings{URDMU_zh,
  title={Dual memory units with uncertainty regulation for weakly supervised video anomaly detection},
  author={Zhou, Hang and Yu, Junqing and Yang, Wei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}

@inproceedings{li2023stprivacy,
  title={STPrivacy: Spatio-temporal privacy-preserving action recognition},
  author={Li, Ming and Xu, Xiangyu and Fan, Hehe and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

## License

This project is released under the MIT License.

## Acknowledgments

This implementation is based on:
- UR-DMU: Dual Memory Units with Uncertainty Regulation
- STPrivacy: Spatio-Temporal Privacy-preserving Action Recognition

