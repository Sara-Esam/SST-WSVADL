# Refactoring Notes for SST-WSVADL

This directory contains a simplified, refactored structure for the two-stage video anomaly detection framework. The current implementation provides a clean structure and simplified training script, but references the original codebase for core functionality.

## Current Status

The refactored version includes:
- ✅ Clean project structure
- ✅ Simplified configuration management
- ✅ Simplified training script (structure)
- ✅ README documentation
- ⚠️ References original codebase for models and utilities

## Full Refactoring Checklist

To create a fully standalone, publishable version, the following components need to be copied/refactored:

### 1. Models (`models/`)
- [ ] `urdmu.py` - WSAD model from `UR-DMU/model.py`
  - Requires: `Memory_Unit`, `Transformer`, `Temporal`, `ADCLS_head`
- [ ] `stpvad.py` - WSVAD_STP model from `UR-DMU/model.py`
  - Requires: `CrossAttentionBlock`, `Memory_Unit`, `Transformer`
- [ ] `memory.py` - Memory_Unit from `UR-DMU/memory.py`
- [ ] `transformer.py` - Transformer from `UR-DMU/translayer.py`
- [ ] `cross_attention.py` - CrossAttentionBlock from `UR-DMU/cross_attention.py`
- [ ] Note: STPrivacy model is already in a separate package (`stprivacy`)

### 2. Data Loaders (`data/`)
- [ ] `datasets.py` - Dataset classes from `UR-DMU/video_segment_loader.py`
  - UCFCrime, XDViolence, MSAD classes

### 3. Loss Functions (`losses/`)
- [ ] `vad_loss.py` - AD_Loss and STPVAD_loss from `UR-DMU/train.py`

### 4. Utilities (`utils/`)
- [ ] `utils.py` - Utility functions from `UR-DMU/utils.py`
  - set_seed, save_best_record, map_topk_snippets_to_frames, etc.

### 5. Evaluation (`eval/`)
- [ ] `test.py` - Test functions from `UR-DMU/ucf_test.py`

### 6. Training Script
- [ ] Complete `two_stage_train` function with full frame extraction
- [ ] Full video processing pipeline
- [ ] Complete evaluation integration

## Simplified Features

The refactored version removes experimental features to keep it simple:
- ❌ RGB-Thermal fusion
- ❌ Motion-based pruning variants
- ❌ Supervised training mode
- ❌ Action recognition training
- ❌ Multi-k sampling
- ❌ Complex loss variants

Kept core features:
- ✅ Two-stage training (UR-DMU + STPVAD)
- ✅ Cross-attention support
- ✅ Basic token pruning
- ✅ Standard loss functions

## Usage

For now, use this as a reference structure. To run:

1. Ensure original UR-DMU and STPrivacy codebases are accessible
2. Update paths in the training script
3. Run: `python train.py --video_root /path/to/videos --test_file /path/to/gt.npy ...`

## Next Steps

1. Copy core model files into `models/`
2. Copy dataset loaders into `data/`
3. Copy utility functions into `utils/`
4. Complete the training script with full implementation
5. Update imports to use local modules
6. Test end-to-end training
7. Add comprehensive documentation



