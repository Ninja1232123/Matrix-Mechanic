"""
Native Quantized Training Module

Train with quantized weights AND quantized gradients.
Instead of training fp32 and hoping it survives quantization,
train natively in quantized space.

Key insight: quantized gradients aren't zero, they're just
snapped to the nearest representable value.
"""

import torch
from typing import Optional

MODULE_INFO = {
    "name": "Native Quantized Training",
    "description": "Train with 4-bit weights and gradients throughout",
    "version": "1.0.0",
    "author": "experimental",

    "config_schema": {
        "enabled": {
            "type": "bool",
            "default": False,
            "description": "Enable native quantized training"
        },
        "weight_bits": {
            "type": "select",
            "options": ["2", "3", "4"],
            "default": "4",
            "description": "Bits for weight quantization"
        },
        "grad_bits": {
            "type": "select",
            "options": ["2", "3", "4", "8"],
            "default": "4",
            "description": "Bits for gradient quantization"
        },
        "accumulation_threshold": {
            "type": "float",
            "default": 0.05,
            "min": 0.001,
            "max": 1.0,
            "description": "Gradient accumulation threshold before weight update"
        },
        "apply_to": {
            "type": "select",
            "options": ["lora_only", "all_trainable"],
            "default": "lora_only",
            "description": "Which parameters to quantize"
        }
    }
}

# LUTs for different bit widths
WEIGHT_LUTS = {
    "2": torch.tensor([-0.5, 0.0, 0.0, 0.5]),
    "3": torch.tensor([-0.5, -0.2, -0.05, 0.0, 0.0, 0.05, 0.2, 0.5]),
    "4": torch.tensor([
        -0.50, -0.30, -0.18, -0.10, -0.05, -0.02, -0.005, 0.0,
         0.005, 0.02,  0.05,  0.10,  0.18,  0.30,  0.50, 1.0
    ])
}

GRAD_LUTS = {
    "2": torch.tensor([-0.1, 0.0, 0.0, 0.1]),
    "3": torch.tensor([-0.1, -0.02, -0.005, 0.0, 0.0, 0.005, 0.02, 0.1]),
    "4": torch.tensor([
        -0.1, -0.05, -0.02, -0.01, -0.005, -0.002, -0.001, 0.0,
         0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2
    ]),
    "8": torch.linspace(-0.2, 0.2, 256)
}

# Per-parameter accumulators (stored globally for persistence across steps)
_accumulators = {}


def quantize_to_lut(x: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Snap tensor to nearest LUT values."""
    x_flat = x.reshape(-1, 1)
    lut_dev = lut.to(x.device).reshape(1, -1)
    distances = torch.abs(x_flat - lut_dev)
    indices = torch.argmin(distances, dim=1)
    return lut.to(x.device)[indices].reshape(x.shape)


def register_routes(app, logger=None):
    """Register API endpoints for this module."""

    @app.route("/api/native_quant/status")
    def native_quant_status():
        return {
            "status": "ok",
            "module": MODULE_INFO["name"],
            "accumulators_tracked": len(_accumulators)
        }

    @app.route("/api/native_quant/reset_accumulators", methods=["POST"])
    def reset_accumulators():
        global _accumulators
        _accumulators = {}
        return {"status": "reset", "message": "Accumulators cleared"}

    if logger:
        logger.info("Native Quant routes registered")


def training_hook(
    model,
    batch,
    step: int,
    config: dict,
    **kwargs
) -> Optional[dict]:
    """
    Apply quantized gradient updates after backward pass.

    Called after loss.backward() but before optimizer.step().
    """
    if not config.get("enabled", False):
        return None

    global _accumulators

    weight_lut = WEIGHT_LUTS[config.get("weight_bits", "4")]
    grad_lut = GRAD_LUTS[config.get("grad_bits", "4")]
    threshold = config.get("accumulation_threshold", 0.05)
    apply_to = config.get("apply_to", "lora_only")

    total_updates = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            # Filter by apply_to setting
            if apply_to == "lora_only" and "lora" not in name.lower():
                continue

            # Initialize accumulator if needed
            if name not in _accumulators:
                _accumulators[name] = torch.zeros_like(param.data)

            # Quantize gradient
            q_grad = quantize_to_lut(param.grad, grad_lut.to(param.device))

            # Accumulate
            _accumulators[name] += q_grad

            # Check threshold
            exceed_mask = torch.abs(_accumulators[name]) >= threshold

            if exceed_mask.any():
                # Compute update direction
                updates = -torch.sign(_accumulators[name][exceed_mask])

                # Quantize current weights
                q_weights = quantize_to_lut(param.data, weight_lut.to(param.device))

                # Find current LUT indices and move them
                # (simplified: just add small delta in update direction)
                delta = updates * weight_lut[1].item()  # smallest positive LUT step
                param.data[exceed_mask] = quantize_to_lut(
                    q_weights[exceed_mask] + delta,
                    weight_lut.to(param.device)
                )

                # Reset accumulators where we updated
                _accumulators[name][exceed_mask] = 0

                total_updates += exceed_mask.sum().item()

    # Return stats for logging
    return {
        "native_quant_updates": total_updates,
        "accumulators_tracked": len(_accumulators)
    }
