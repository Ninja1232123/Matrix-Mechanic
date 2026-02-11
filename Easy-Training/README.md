###AI TRAINING FOR DUMDUMS

This directory contains the original AI-Training-For-DumDums codebase components that were not part of the π/2 quantization focus.

## What's Here

### App/
The original massive Flask application (5,110 lines) with:
- Multichannel encoding system
- LoRA/QLoRA advanced training
- Model comparison tools
- Full web UI with all features
- 50+ Python modules
- Complete frontend (189KB HTML, 95KB JS)

### data/
Original data generation and encoding scripts:
- Multichannel encoder
- Curriculum generators
- Dataset partitioners
- Various training data files

### tests/
Original test suite for the full application

## Why Archived?

The main project was refactored to focus specifically on π/2 quantization training. These components represent the base10/multichannel training paradigm and experimental features that aren't needed for the simplified π/2 app.

## What Was Kept?

The following were extracted and simplified for the π/2 app:
- **pi_universal_encoder.py** → Multi-modal π/2 encoder
- **pi_quantizer.py** → π-based quantization
- **autonomous_mind.py** → Continuous thinking entity
- **thought_threshold.py** → Thought crystallization detector
- **pi2_trainer.py** → Standalone π/2 trainer

## Reference

You can still reference these files if you need to:
- Understand the multichannel encoding approach
- Reuse LoRA/QLoRA training code
- Extract comparison functionality
- Study the Union Break experiments

## Restoration

If you need to restore any of this functionality:
1. Copy the needed modules from `App/` to your new structure
2. Adapt imports and dependencies
3. Integrate into the new pi2_app

---

**Archived on:** 2025-12-20
**Reason:** Refactoring to focus on π/2 quantization only
