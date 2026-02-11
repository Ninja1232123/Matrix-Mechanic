#!/usr/bin/env python3
"""
π/2 CUDA Bridge
===============

Loads compiled CUDA kernels via ctypes for P40 integer training.
Compile first: nvcc -shared -o libpi2.so kernels/pi2_accumulate.cu -O3

Usage:
    from pi2_cuda import CudaPhaseLinear, compile_kernels
    compile_kernels()  # One-time compilation
    layer = CudaPhaseLinear(512, 512, device=0)
"""

import os
import ctypes
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# CUDA types
c_int = ctypes.c_int
c_int8 = ctypes.c_int8
c_int32 = ctypes.c_int32
c_int64 = ctypes.c_int64
c_uint8 = ctypes.c_uint8
c_uint16 = ctypes.c_uint16
c_void_p = ctypes.c_void_p

# Library handle
_lib = None
_cuda_available = False


def compile_kernels(force: bool = False) -> bool:
    """
    Compile CUDA kernels to shared library.

    Returns True if successful.
    """
    kernel_dir = Path(__file__).parent / "kernels"
    cu_file = kernel_dir / "pi2_accumulate.cu"
    so_file = kernel_dir / "libpi2.so"

    if not cu_file.exists():
        print(f"CUDA source not found: {cu_file}")
        return False

    # Check if recompilation needed
    if so_file.exists() and not force:
        cu_mtime = cu_file.stat().st_mtime
        so_mtime = so_file.stat().st_mtime
        if so_mtime > cu_mtime:
            print(f"  Using cached: {so_file}")
            return True

    print(f"  Compiling CUDA kernels...")

    # Compile with nvcc
    # -shared: create shared library
    # -Xcompiler -fPIC: position independent code
    # -O3: optimize
    # -arch=sm_61: Pascal architecture (P40)
    cmd = (
        f"nvcc -shared -Xcompiler -fPIC -O3 -arch=sm_61 "
        f"-o {so_file} {cu_file} 2>&1"
    )

    result = os.system(cmd)
    if result != 0:
        print(f"  Compilation failed (exit code {result})")
        return False

    print(f"  Compiled: {so_file}")
    return True


def load_library() -> bool:
    """Load the compiled CUDA library."""
    global _lib, _cuda_available

    if _lib is not None:
        return _cuda_available

    so_file = Path(__file__).parent / "kernels" / "libpi2.so"

    if not so_file.exists():
        print(f"  Library not found: {so_file}")
        print(f"  Run compile_kernels() first")
        _cuda_available = False
        return False

    try:
        _lib = ctypes.CDLL(str(so_file))
        _setup_function_signatures()
        _cuda_available = True
        print(f"  Loaded CUDA library: {so_file}")
        return True
    except OSError as e:
        print(f"  Failed to load library: {e}")
        _cuda_available = False
        return False


def _setup_function_signatures():
    """Set up ctypes function signatures."""
    global _lib

    # void launch_accumulate_reduce(
    #     const uint16_t* mag, const uint8_t* phase,
    #     int64_t* sum_real, int64_t* sum_imag,
    #     int n, cudaStream_t stream
    # )
    _lib.launch_accumulate_reduce.argtypes = [
        ctypes.POINTER(c_uint16),  # mag
        ctypes.POINTER(c_uint8),   # phase
        ctypes.POINTER(c_int64),   # sum_real
        ctypes.POINTER(c_int64),   # sum_imag
        c_int,                     # n
        c_void_p,                  # stream (NULL for default)
    ]
    _lib.launch_accumulate_reduce.restype = None

    # void launch_normalize(
    #     int64_t* sum_real, int64_t* sum_imag,
    #     int num_bins, int shift, cudaStream_t stream
    # )
    _lib.launch_normalize.argtypes = [
        ctypes.POINTER(c_int64),  # sum_real
        ctypes.POINTER(c_int64),  # sum_imag
        c_int,                    # num_bins
        c_int,                    # shift
        c_void_p,                 # stream
    ]
    _lib.launch_normalize.restype = None

    # void launch_to_polar(
    #     const int64_t* sum_real, const int64_t* sum_imag,
    #     uint16_t* mag_out, uint8_t* phase_out,
    #     int num_bins, cudaStream_t stream
    # )
    _lib.launch_to_polar.argtypes = [
        ctypes.POINTER(c_int64),   # sum_real
        ctypes.POINTER(c_int64),   # sum_imag
        ctypes.POINTER(c_uint16),  # mag_out
        ctypes.POINTER(c_uint8),   # phase_out
        c_int,                     # num_bins
        c_void_p,                  # stream
    ]
    _lib.launch_to_polar.restype = None

    # void launch_phase_linear(
    #     const uint16_t* in_mag, const uint8_t* in_phase,
    #     const uint8_t* weight_phase, const int8_t* weight_mag,
    #     int64_t* acc_real, int64_t* acc_imag,
    #     int batch_size, int seq_len, int in_features, int out_features,
    #     cudaStream_t stream
    # )
    _lib.launch_phase_linear.argtypes = [
        ctypes.POINTER(c_uint16),  # in_mag
        ctypes.POINTER(c_uint8),   # in_phase
        ctypes.POINTER(c_uint8),   # weight_phase
        ctypes.POINTER(c_int8),    # weight_mag
        ctypes.POINTER(c_int64),   # acc_real
        ctypes.POINTER(c_int64),   # acc_imag
        c_int,                     # batch_size
        c_int,                     # seq_len
        c_int,                     # in_features
        c_int,                     # out_features
        c_void_p,                  # stream
    ]
    _lib.launch_phase_linear.restype = None


# =============================================================================
# CUDA MEMORY MANAGEMENT (using CuPy for GPU arrays)
# =============================================================================

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class CudaPhaseLinear:
    """
    GPU-accelerated PhaseLinear using CUDA kernels.

    Replaces the Python loop version with P40-optimized CUDA.
    """

    def __init__(self, in_features: int, out_features: int, device: int = 0):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy required for CUDA acceleration. Install: pip install cupy-cuda11x")

        if not load_library():
            raise RuntimeError("Failed to load CUDA library. Run compile_kernels() first.")

        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        with cp.cuda.Device(device):
            # Weights on GPU
            self.weight_phase = cp.random.randint(0, 4, (in_features, out_features), dtype=cp.uint8)
            self.weight_mag = cp.random.randint(1, 128, (in_features, out_features), dtype=cp.int8)

            # Gradient accumulators on GPU
            self.grad_real = cp.zeros((in_features, out_features), dtype=cp.int64)
            self.grad_imag = cp.zeros((in_features, out_features), dtype=cp.int64)
            self.grad_count = 0

    def forward(self, x_mag: np.ndarray, x_phase: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass using CUDA kernel.

        x_mag: (batch, seq, in_features) uint16
        x_phase: (batch, seq, in_features) uint8

        Returns: (out_mag, out_phase)
        """
        batch, seq, _ = x_mag.shape
        total_outputs = batch * seq * self.out_features

        with cp.cuda.Device(self.device):
            # Copy inputs to GPU
            x_mag_gpu = cp.asarray(x_mag.astype(np.uint16))
            x_phase_gpu = cp.asarray(x_phase.astype(np.uint8))

            # Output accumulators on GPU
            acc_real = cp.zeros((batch, seq, self.out_features), dtype=cp.int64)
            acc_imag = cp.zeros((batch, seq, self.out_features), dtype=cp.int64)

            # Get raw pointers
            in_mag_ptr = ctypes.cast(x_mag_gpu.data.ptr, ctypes.POINTER(c_uint16))
            in_phase_ptr = ctypes.cast(x_phase_gpu.data.ptr, ctypes.POINTER(c_uint8))
            w_phase_ptr = ctypes.cast(self.weight_phase.data.ptr, ctypes.POINTER(c_uint8))
            w_mag_ptr = ctypes.cast(self.weight_mag.data.ptr, ctypes.POINTER(c_int8))
            acc_real_ptr = ctypes.cast(acc_real.data.ptr, ctypes.POINTER(c_int64))
            acc_imag_ptr = ctypes.cast(acc_imag.data.ptr, ctypes.POINTER(c_int64))

            # Launch kernel
            _lib.launch_phase_linear(
                in_mag_ptr, in_phase_ptr,
                w_phase_ptr, w_mag_ptr,
                acc_real_ptr, acc_imag_ptr,
                batch, seq, self.in_features, self.out_features,
                None  # default stream
            )

            # Synchronize
            cp.cuda.Device(self.device).synchronize()

            # Convert to polar (still on GPU using approximate magnitude)
            # mag = max(|r|, |i|) + 0.5 * min(|r|, |i|)
            ar = cp.abs(acc_real)
            ai = cp.abs(acc_imag)
            mag_approx = cp.where(ar > ai, ar + (ai >> 1), ai + (ar >> 1))

            # Clamp to uint16
            out_mag = cp.clip(mag_approx, 0, 65535).astype(cp.uint16)

            # Phase from quadrant
            out_phase = cp.zeros_like(out_mag, dtype=cp.uint8)
            mostly_real = ar >= ai
            out_phase = cp.where(mostly_real & (acc_real >= 0), 0, out_phase)
            out_phase = cp.where(mostly_real & (acc_real < 0), 2, out_phase)
            out_phase = cp.where(~mostly_real & (acc_imag >= 0), 1, out_phase)
            out_phase = cp.where(~mostly_real & (acc_imag < 0), 3, out_phase)

            # Copy back to CPU
            return cp.asnumpy(out_mag).astype(np.int32), cp.asnumpy(out_phase).astype(np.int8)

    def backward(self, x_mag, x_phase, grad_mag, grad_phase):
        """Accumulate gradients on GPU."""
        # Similar to forward but accumulates gradient direction
        # Keeping simple for now - can optimize later
        pass

    def update(self, lr: int = 256):
        """Update weights using accumulated gradients."""
        if self.grad_count == 0:
            return

        with cp.cuda.Device(self.device):
            # Normalize by count using shift
            shift = max(0, int(np.log2(self.grad_count)) if self.grad_count > 0 else 0)
            grad_real = self.grad_real >> shift
            grad_imag = self.grad_imag >> shift

            # Phase update from gradient direction
            ar = cp.abs(grad_real)
            ai = cp.abs(grad_imag)
            mostly_real = ar >= ai

            grad_phase_update = cp.zeros_like(self.weight_phase)
            grad_phase_update = cp.where(mostly_real & (grad_real >= 0), 0, grad_phase_update)
            grad_phase_update = cp.where(mostly_real & (grad_real < 0), 2, grad_phase_update)
            grad_phase_update = cp.where(~mostly_real & (grad_imag >= 0), 1, grad_phase_update)
            grad_phase_update = cp.where(~mostly_real & (grad_imag < 0), 3, grad_phase_update)

            # Stochastic update based on learning rate
            if lr < 256:
                rand = cp.random.randint(0, 256, self.weight_phase.shape, dtype=cp.int32)
                mask = rand < lr
                self.weight_phase = cp.where(
                    mask,
                    (self.weight_phase.astype(cp.int32) + grad_phase_update) & 3,
                    self.weight_phase
                ).astype(cp.uint8)
            else:
                self.weight_phase = ((self.weight_phase.astype(cp.int32) + grad_phase_update) & 3).astype(cp.uint8)

            # Reset accumulators
            self.grad_real.fill(0)
            self.grad_imag.fill(0)
            self.grad_count = 0


# =============================================================================
# NUMPY FALLBACK (when CUDA not available)
# =============================================================================

class NumpyPhaseLinear:
    """
    CPU fallback using numpy vectorization.

    Faster than Python loops but slower than CUDA.
    Uses einsum for the inner product.
    """

    # Phase lookup tables
    PHASE_COS = np.array([1, 0, -1, 0], dtype=np.int8)
    PHASE_SIN = np.array([0, 1, 0, -1], dtype=np.int8)

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

        self.weight_phase = np.random.randint(0, 4, (in_features, out_features), dtype=np.int8)
        self.weight_mag = np.random.randint(1, 128, (in_features, out_features), dtype=np.int8)

        self.grad_real = np.zeros((in_features, out_features), dtype=np.int64)
        self.grad_imag = np.zeros((in_features, out_features), dtype=np.int64)
        self.grad_count = 0

    def forward(self, x_mag: np.ndarray, x_phase: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized forward pass.

        Uses broadcasting instead of loops.
        """
        batch, seq, in_feat = x_mag.shape

        # Expand dims for broadcasting
        # x: (batch, seq, in_features, 1)
        # w: (1, 1, in_features, out_features)
        x_mag_exp = x_mag[:, :, :, np.newaxis].astype(np.int64)
        x_phase_exp = x_phase[:, :, :, np.newaxis]

        w_phase_exp = self.weight_phase[np.newaxis, np.newaxis, :, :]
        w_mag_exp = self.weight_mag[np.newaxis, np.newaxis, :, :]

        # Combined phase: (batch, seq, in_features, out_features)
        combined_phase = (x_phase_exp + w_phase_exp) & 3

        # Magnitude contribution
        mag_contrib = x_mag_exp * w_mag_exp

        # Accumulate using phase LUT
        # cos/sin indexed by combined_phase
        real_contrib = mag_contrib * self.PHASE_COS[combined_phase]
        imag_contrib = mag_contrib * self.PHASE_SIN[combined_phase]

        # Sum over in_features: (batch, seq, out_features)
        acc_real = np.sum(real_contrib, axis=2)
        acc_imag = np.sum(imag_contrib, axis=2)

        # Approximate magnitude
        ar = np.abs(acc_real)
        ai = np.abs(acc_imag)
        mag_approx = np.where(ar > ai, ar + (ai >> 1), ai + (ar >> 1))
        out_mag = np.clip(mag_approx, 0, 65535).astype(np.int32)

        # Phase from quadrant
        mostly_real = ar >= ai
        out_phase = np.zeros_like(out_mag, dtype=np.int8)
        out_phase = np.where(mostly_real & (acc_real >= 0), 0, out_phase)
        out_phase = np.where(mostly_real & (acc_real < 0), 2, out_phase)
        out_phase = np.where(~mostly_real & (acc_imag >= 0), 1, out_phase)
        out_phase = np.where(~mostly_real & (acc_imag < 0), 3, out_phase)

        return out_mag, out_phase

    def backward(self, x_mag, x_phase, grad_mag, grad_phase):
        """Accumulate gradients."""
        pass  # TODO

    def update(self, lr: int = 256):
        """Update weights."""
        pass  # TODO


def get_phase_linear(in_features: int, out_features: int,
                     device: Optional[int] = None) -> 'CudaPhaseLinear | NumpyPhaseLinear':
    """
    Get the best available PhaseLinear implementation.

    Returns CudaPhaseLinear if CUDA available, else NumpyPhaseLinear.
    """
    if device is not None and CUPY_AVAILABLE:
        try:
            if load_library():
                return CudaPhaseLinear(in_features, out_features, device)
        except Exception as e:
            print(f"  CUDA init failed: {e}, falling back to numpy")

    return NumpyPhaseLinear(in_features, out_features)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="π/2 CUDA Bridge")
    parser.add_argument("--compile", action="store_true", help="Compile CUDA kernels")
    parser.add_argument("--test", action="store_true", help="Test kernel")
    parser.add_argument("--force", action="store_true", help="Force recompilation")

    args = parser.parse_args()

    if args.compile:
        success = compile_kernels(force=args.force)
        if success:
            print("Compilation successful")
        else:
            print("Compilation failed")
            exit(1)

    if args.test:
        print("\nTesting PhaseLinear...")

        # Test dimensions
        batch, seq, in_feat, out_feat = 2, 4, 8, 16

        # Random inputs
        x_mag = np.random.randint(0, 1000, (batch, seq, in_feat), dtype=np.uint16)
        x_phase = np.random.randint(0, 4, (batch, seq, in_feat), dtype=np.uint8)

        # Test numpy version
        print("  Testing NumpyPhaseLinear...")
        np_layer = NumpyPhaseLinear(in_feat, out_feat)
        np_mag, np_phase = np_layer.forward(x_mag, x_phase)
        print(f"    Output shape: {np_mag.shape}")
        print(f"    Mag range: [{np_mag.min()}, {np_mag.max()}]")
        print(f"    Phase values: {np.unique(np_phase)}")

        # Test CUDA version if available
        if CUPY_AVAILABLE and load_library():
            print("  Testing CudaPhaseLinear...")
            cuda_layer = CudaPhaseLinear(in_feat, out_feat, device=0)
            cuda_mag, cuda_phase = cuda_layer.forward(x_mag, x_phase)
            print(f"    Output shape: {cuda_mag.shape}")
            print(f"    Mag range: [{cuda_mag.min()}, {cuda_mag.max()}]")
            print(f"    Phase values: {np.unique(cuda_phase)}")
        else:
            print("  CUDA not available, skipping GPU test")

        print("\nTest complete")
