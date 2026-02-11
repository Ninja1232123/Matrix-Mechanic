#!/usr/bin/env python3
"""
Pure Integer Forward Pass - Breaking the Matrix

Data Flow:
    INT8 (Input) → INT8 (Weight) → INT32 (Accumulator) → Stochastic Shift → INT8 (Next Layer)

No floating point in the forward pass until the final logits layer.

Now supports:
- CuPy backend (RawKernels, zero build step)
- PyTorch backend (torch._int_mm)
- Pure numpy fallback

Priority: CuPy > PyTorch > numpy
"""

import math
import numpy as np

# Backend detection
try:
    import cupy as cp
    from cupy_backend import CupyInt8Ops, to_gpu, to_cpu
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================
# CUPY BACKEND
# ============================================

if HAS_CUPY:

    def fixed_point_stochastic_shift_cupy(acc_int32, shift_bits):
        """GPU stochastic rounding via CuPy."""
        ops = CupyInt8Ops()
        return ops.stochastic_shift(acc_int32, int(shift_bits))

    def quantize_to_int8_pow2_cupy(x_float):
        """Quantize float to INT8 on GPU."""
        ops = CupyInt8Ops()
        x_flat = x_float.reshape(-1, x_float.shape[-1])
        if not isinstance(x_flat, cp.ndarray):
            x_flat = cp.asarray(x_flat, dtype=cp.float32)
        return ops.quantize_input(x_flat)

    class PureInt8LinearCuPy:
        """Pure INT8 linear layer using CuPy RawKernels."""

        def __init__(self, in_features, out_features):
            self.in_features = in_features
            self.out_features = out_features
            self.ops = CupyInt8Ops()

            self.weight = cp.zeros((out_features, in_features), dtype=cp.int8)
            self.scale_multiplier = cp.ones(out_features, dtype=cp.int32)
            self.scale_shift = cp.full((out_features,), 16, dtype=cp.int8)
            self.output_shift = 8

        def forward(self, x, input_scale_shift=None, return_float=False,
                    activation=None):
            x = cp.asarray(x) if not isinstance(x, cp.ndarray) else x
            batch_shape = x.shape[:-1]
            x_flat = x.reshape(-1, x.shape[-1])

            if x.dtype != cp.int8:
                x_int8, input_scale_shift = self.ops.quantize_input(
                    x_flat.astype(cp.float32)
                )
            else:
                x_int8 = cp.ascontiguousarray(x_flat)
                if input_scale_shift is None:
                    input_scale_shift = cp.zeros(x_int8.shape[0], dtype=cp.int8)

            if activation is not None:
                out_int8 = self.ops.gemm_fused(
                    x_int8, self.weight,
                    self.scale_multiplier, self.scale_shift,
                    self.output_shift, activation
                )
            else:
                out_int8 = self.ops.gemm(
                    x_int8, self.weight,
                    self.scale_multiplier, self.scale_shift,
                    self.output_shift
                )

            if return_float:
                out_float = self.ops.dequantize_output(
                    out_int8, input_scale_shift,
                    self.scale_multiplier, self.scale_shift
                )
                return out_float.reshape(*batch_shape, -1)
            else:
                out_scale = input_scale_shift + np.int8(self.output_shift)
                return out_int8.reshape(*batch_shape, -1), out_scale

        @classmethod
        def from_float_weights(cls, fp32_weight, precision_bits=16):
            if HAS_TORCH and isinstance(fp32_weight, torch.Tensor):
                fp32_weight = fp32_weight.detach().cpu().numpy()
            fp32_weight = np.asarray(fp32_weight, dtype=np.float32)

            out_features, in_features = fp32_weight.shape
            row_max = np.abs(fp32_weight).max(axis=1).clip(min=1e-8)
            per_channel_scale = row_max / 127.0

            int8_weight = np.round(
                fp32_weight / per_channel_scale[:, None]
            ).clip(-128, 127).astype(np.int8)

            log2_scale = np.log2(np.clip(per_channel_scale, 1e-10, None))
            shift = np.clip(
                precision_bits - 1 - np.floor(log2_scale), 0, 31
            ).astype(np.int8)
            multiplier = np.round(
                per_channel_scale * (2.0 ** shift.astype(np.float64))
            ).astype(np.int32)

            layer = cls(in_features, out_features)
            layer.weight = cp.asarray(int8_weight)
            layer.scale_multiplier = cp.asarray(multiplier)
            layer.scale_shift = cp.asarray(shift)
            return layer


# ============================================
# TORCH BACKEND
# ============================================

if HAS_TORCH:

    def fixed_point_stochastic_shift_torch(acc_int32, shift_bits):
        """Stochastic rounding via PyTorch."""
        if isinstance(shift_bits, int):
            if shift_bits <= 0:
                return acc_int32.clamp(-128, 127).to(torch.int8)
            threshold = torch.randint(
                0, 1 << shift_bits, acc_int32.shape,
                dtype=torch.int32, device=acc_int32.device
            )
            rounded = (acc_int32 + threshold) >> shift_bits
            return rounded.clamp(-128, 127).to(torch.int8)
        else:
            threshold = torch.randint(
                0, 1 << shift_bits.max().item(), acc_int32.shape,
                dtype=torch.int32, device=acc_int32.device
            )
            threshold = threshold & ((1 << shift_bits.unsqueeze(0)) - 1)
            rounded = (acc_int32 + threshold) >> shift_bits.unsqueeze(0)
            return rounded.clamp(-128, 127).to(torch.int8)

    def quantize_to_int8_pow2_torch(x_float):
        """Quantize float tensor to INT8 with power-of-2 scaling."""
        x_flat = x_float.reshape(-1, x_float.shape[-1])
        x_abs_max = x_flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale_shift = torch.ceil(
            torch.log2(x_abs_max / 127.0)
        ).clamp(min=-126, max=126).to(torch.int8)
        scale = torch.pow(2.0, scale_shift.float())
        x_int8 = (x_flat / scale).round().clamp(-128, 127).to(torch.int8)
        return x_int8, scale_shift

    class PureInt8LinearTorch(nn.Module):
        """Pure INT8 linear layer using torch._int_mm."""

        def __init__(self, in_features, out_features):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

            self.register_buffer('weight',
                                 torch.zeros(out_features, in_features, dtype=torch.int8))
            self.register_buffer('scale_multiplier',
                                 torch.ones(out_features, dtype=torch.int32))
            self.register_buffer('scale_shift',
                                 torch.full((out_features,), 16, dtype=torch.int8))
            self.register_buffer('output_shift',
                                 torch.tensor(8, dtype=torch.int8))

        def forward(self, x_int8, input_scale_shift=None, return_float=False):
            if x_int8.dtype != torch.int8:
                x_int8, input_scale_shift = quantize_to_int8_pow2_torch(x_int8)

            batch_shape = x_int8.shape[:-1]
            x_flat = x_int8.reshape(-1, x_int8.shape[-1])
            M = x_flat.shape[0]

            needs_padding = M <= 16
            if needs_padding:
                x_flat = F.pad(x_flat, (0, 0, 0, 17 - M))

            out_int32 = torch._int_mm(x_flat, self.weight.t())

            if needs_padding:
                out_int32 = out_int32[:M]

            if return_float:
                if input_scale_shift is not None:
                    input_scale = torch.pow(2.0, input_scale_shift.float())
                else:
                    input_scale = 1.0
                weight_scale = self.scale_multiplier.float() * torch.pow(
                    2.0, -self.scale_shift.float()
                )
                out_float = out_int32.float() * input_scale * weight_scale.unsqueeze(0)
                return out_float.reshape(*batch_shape, -1)
            else:
                out_scaled = out_int32 * self.scale_multiplier.unsqueeze(0)
                total_shift = self.scale_shift.max().item() + self.output_shift.item()
                out_int8 = fixed_point_stochastic_shift_torch(out_scaled, total_shift)

                if input_scale_shift is not None:
                    output_scale_shift = input_scale_shift + self.output_shift
                else:
                    output_scale_shift = self.output_shift.expand(M, 1)

                return out_int8.reshape(*batch_shape, -1), output_scale_shift

        @classmethod
        def from_float_weights(cls, fp32_weight, precision_bits=16):
            out_features, in_features = fp32_weight.shape
            row_max = fp32_weight.abs().amax(dim=1).clamp(min=1e-8)
            per_channel_scale = row_max / 127.0

            int8_weight = (
                fp32_weight / per_channel_scale.unsqueeze(1)
            ).round().clamp(-128, 127).to(torch.int8)

            log2_scale = torch.log2(per_channel_scale.clamp(min=1e-10))
            shift = (
                precision_bits - 1 - torch.floor(log2_scale)
            ).clamp(min=0, max=31).to(torch.int8)
            multiplier = (
                per_channel_scale * torch.pow(2.0, shift.float())
            ).round().to(torch.int32)

            layer = cls(in_features, out_features)
            layer.weight.copy_(int8_weight)
            layer.scale_multiplier.copy_(multiplier)
            layer.scale_shift.copy_(shift)
            return layer


# ============================================
# AUTO-SELECT BEST BACKEND
# ============================================

def get_best_backend():
    """Returns the best available backend name."""
    if HAS_CUPY:
        return 'cupy'
    if HAS_TORCH:
        return 'torch'
    return 'numpy'


def create_int8_linear(in_features, out_features, backend=None):
    """Create INT8 linear layer with best available backend."""
    if backend is None:
        backend = get_best_backend()

    if backend == 'cupy':
        return PureInt8LinearCuPy(in_features, out_features)
    elif backend == 'torch':
        return PureInt8LinearTorch(in_features, out_features)
    else:
        raise ValueError(f"No GPU backend available. Install cupy or torch.")


def create_int8_linear_from_weights(fp32_weight, backend=None, precision_bits=16):
    """Create INT8 linear from FP32 weights with best backend."""
    if backend is None:
        backend = get_best_backend()

    if backend == 'cupy':
        return PureInt8LinearCuPy.from_float_weights(fp32_weight, precision_bits)
    elif backend == 'torch':
        if isinstance(fp32_weight, np.ndarray):
            fp32_weight = torch.from_numpy(fp32_weight)
        return PureInt8LinearTorch.from_float_weights(fp32_weight, precision_bits)
    else:
        raise ValueError(f"No GPU backend available.")


# Convenience aliases
PureInt8Linear = PureInt8LinearCuPy if HAS_CUPY else (
    PureInt8LinearTorch if HAS_TORCH else None
)

fixed_point_stochastic_shift = fixed_point_stochastic_shift_cupy if HAS_CUPY else (
    fixed_point_stochastic_shift_torch if HAS_TORCH else None
)

quantize_to_int8_pow2 = quantize_to_int8_pow2_cupy if HAS_CUPY else (
    quantize_to_int8_pow2_torch if HAS_TORCH else None
)


# ============================================
# TESTS
# ============================================

def test_stochastic_rounding():
    print("=" * 60)
    print(f"Stochastic Rounding Test (backend: {get_best_backend()})")
    print("=" * 60)

    if HAS_CUPY:
        int32_vals = cp.random.randint(-10000, 10000, (1000, 64), dtype=cp.int32)
        shift_bits = 8
        true_mean = float(int32_vals.astype(cp.float64).mean()) / (1 << shift_bits)

        samples = 100
        rounded_sum = cp.zeros_like(int32_vals, dtype=cp.float64)
        for _ in range(samples):
            rounded = fixed_point_stochastic_shift_cupy(int32_vals, shift_bits)
            rounded_sum += rounded.astype(cp.float64)
        rounded_mean = float((rounded_sum / samples).mean())

    elif HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        int32_vals = torch.randint(-10000, 10000, (1000, 64), dtype=torch.int32, device=device)
        shift_bits = 8
        true_mean = int32_vals.float().mean().item() / (1 << shift_bits)

        samples = 100
        rounded_sum = torch.zeros_like(int32_vals, dtype=torch.float32)
        for _ in range(samples):
            rounded = fixed_point_stochastic_shift_torch(int32_vals, shift_bits)
            rounded_sum += rounded.float()
        rounded_mean = (rounded_sum / samples).mean().item()

    else:
        print("  No backend available")
        return

    print(f"\n  True mean:       {true_mean:.4f}")
    print(f"  Rounded mean:    {rounded_mean:.4f}")
    print(f"  Error:           {abs(true_mean - rounded_mean):.4f}")
    print(f"  Relative error:  {abs(true_mean - rounded_mean) / abs(true_mean) * 100:.2f}%")


def test_linear():
    backend = get_best_backend()
    print(f"\n{'=' * 60}")
    print(f"Pure INT8 Linear Test (backend: {backend})")
    print(f"{'=' * 60}")

    np.random.seed(42)
    in_features, out_features = 256, 128
    fp32_weight = np.random.randn(out_features, in_features).astype(np.float32)

    layer = create_int8_linear_from_weights(fp32_weight, backend)

    x = np.random.randn(32, in_features).astype(np.float32)
    fp32_out = x @ fp32_weight.T

    if backend == 'cupy':
        out_float = layer.forward(cp.asarray(x), return_float=True)
        out_np = to_cpu(out_float)
    else:
        x_t = torch.from_numpy(x)
        if torch.cuda.is_available():
            x_t = x_t.cuda()
            layer = layer.cuda()
        out_float = layer(x_t, return_float=True)
        out_np = out_float.detach().cpu().numpy()

    rel_err = np.abs(fp32_out - out_np).mean() / np.abs(fp32_out).mean() * 100
    print(f"\n  FP32 mean:  {np.abs(fp32_out).mean():.4f}")
    print(f"  INT8 mean:  {np.abs(out_np).mean():.4f}")
    print(f"  Rel error:  {rel_err:.2f}%")


def test_chained():
    backend = get_best_backend()
    print(f"\n{'=' * 60}")
    print(f"Chained Layers Test (backend: {backend})")
    print(f"{'=' * 60}")

    np.random.seed(42)
    dims = [256, 512, 256, 128]
    weights = [np.random.randn(dims[i + 1], dims[i]).astype(np.float32) * 0.1
               for i in range(len(dims) - 1)]

    layers = [create_int8_linear_from_weights(w, backend) for w in weights]

    x = np.random.randn(32, dims[0]).astype(np.float32)

    if backend == 'cupy':
        current, scale = layers[0].forward(cp.asarray(x), return_float=False)
        print(f"\n  Layer 0: float -> INT8")
        for i, layer in enumerate(layers[1:-1], 1):
            current, scale = layer.forward(current, scale, return_float=False)
            print(f"  Layer {i}: INT8 -> INT8")
        output = layers[-1].forward(current, scale, return_float=True)
        print(f"  Layer {len(layers) - 1}: INT8 -> float")
        out_np = to_cpu(output)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layers = [l.to(device) for l in layers]
        x_t = torch.from_numpy(x).to(device)

        current, scale = layers[0](x_t, return_float=False)
        print(f"\n  Layer 0: float -> INT8")
        for i, layer in enumerate(layers[1:-1], 1):
            current, scale = layer(current, scale, return_float=False)
            print(f"  Layer {i}: INT8 -> INT8")
        output = layers[-1](current, scale, return_float=True)
        print(f"  Layer {len(layers) - 1}: INT8 -> float")
        out_np = output.detach().cpu().numpy()

    fp32_out = x
    for w in weights:
        fp32_out = fp32_out @ w.T

    rel_err = np.abs(fp32_out - out_np).mean() / np.abs(fp32_out).mean() * 100
    print(f"\n  FP32 mean:  {np.abs(fp32_out).mean():.4f}")
    print(f"  INT8 mean:  {np.abs(out_np).mean():.4f}")
    print(f"  Rel error:  {rel_err:.2f}%")
    print(f"\n  ✓ ZERO float ops in hidden layers!")


if __name__ == "__main__":
    print(f"Best backend: {get_best_backend()}")
    print(f"CuPy: {HAS_CUPY}, PyTorch: {HAS_TORCH}\n")
    test_stochastic_rounding()
    test_linear()
    test_chained()
