"""
QAT LUT Module - Quantization-Aware Training with Look-Up Tables

Periodically pushes weights toward LUT values during training,
so the model learns to be robust to hard quantization.

This is the "soft" approach: train at full precision but show
the model what quantization looks like.
"""

import torch
from typing import Optional

MODULE_INFO = {
    "name": "QAT with LUT",
    "description": "Push weights toward quantization-friendly values during training",
    "version": "1.0.0",

    "config_schema": {
        "enabled": {
            "type": "bool",
            "default": False,
            "description": "Enable QAT"
        },
        "lut_bits": {
            "type": "select",
            "options": ["3", "4"],
            "default": "4",
            "description": "LUT precision (3=8 levels, 4=16 levels)"
        },
        "strength": {
            "type": "float",
            "default": 0.1,
            "min": 0.01,
            "max": 0.5,
            "description": "How strongly to push weights toward LUT (0.1 = 10%)"
        },
        "frequency": {
            "type": "int",
            "default": 10,
            "min": 1,
            "max": 100,
            "description": "Apply every N steps"
        },
        "apply_to": {
            "type": "select",
            "options": ["lora_only", "all_trainable"],
            "default": "lora_only",
            "description": "Which parameters to affect"
        }
    }
}

# K-means optimized LUTs
LUT_8 = torch.tensor([
    -0.0156, -0.0052, -0.0017, -0.0003,
     0.0003,  0.0017,  0.0052,  0.0156
], dtype=torch.float32)

LUT_16 = torch.tensor([
    -0.0200, -0.0120, -0.0075, -0.0045, -0.0025, -0.0012, -0.0005, -0.0001,
     0.0001,  0.0005,  0.0012,  0.0025,  0.0045,  0.0075,  0.0120,  0.0200
], dtype=torch.float32)


def quantize_to_lut(x: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Snap tensor to nearest LUT values."""
    x_flat = x.reshape(-1, 1)
    lut_dev = lut.to(x.device).reshape(1, -1)
    distances = torch.abs(x_flat - lut_dev)
    indices = torch.argmin(distances, dim=1)
    return lut.to(x.device)[indices].reshape(x.shape)


def calc_quantization_error(model, lut: torch.Tensor, apply_to: str = "lora_only") -> float:
    """Calculate average distance from LUT values."""
    total_error = 0
    total_weights = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if apply_to == "lora_only" and "lora" not in name.lower():
            continue

        w = param.data.float()
        q = quantize_to_lut(w, lut.to(w.device))
        error = (w - q).abs().mean().item()
        total_error += error * w.numel()
        total_weights += w.numel()

    return total_error / total_weights if total_weights > 0 else 0


def register_routes(app, logger=None):
    """Register API endpoints."""

    @app.route("/api/qat_lut/error")
    def qat_error():
        # Would need model reference - simplified for now
        return {
            "status": "ok",
            "module": MODULE_INFO["name"],
            "note": "Use training metrics for Q-Error"
        }

    if logger:
        logger.info("QAT LUT routes registered")


def training_hook(
    model,
    batch,
    step: int,
    config: dict,
    **kwargs
) -> Optional[dict]:
    """
    Push weights toward LUT values every N steps.
    """
    if not config.get("enabled", False):
        return None

    frequency = config.get("frequency", 10)
    if step % frequency != 0:
        return None

    strength = config.get("strength", 0.1)
    bits = config.get("lut_bits", "4")
    apply_to = config.get("apply_to", "lora_only")

    lut = LUT_16 if bits == "4" else LUT_8

    # Calculate error before
    error_before = calc_quantization_error(model, lut, apply_to)

    # Push weights toward LUT
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if apply_to == "lora_only" and "lora" not in name.lower():
                continue

            w = param.data
            q = quantize_to_lut(w, lut.to(w.device))
            # Interpolate toward quantized values
            param.data = w + strength * (q - w)

    # Calculate error after
    error_after = calc_quantization_error(model, lut, apply_to)

    return {
        "qat_error_before": error_before,
        "qat_error_after": error_after,
        "qat_reduction": (error_before - error_after) / error_before * 100 if error_before > 0 else 0
    }
