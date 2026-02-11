#!/usr/bin/env python3
"""
CuPy Backend for Xirtam INT8 Kernels

Replaces:
- ctypes .so loading (xirtam_pure.py)
- PyTorch C++ extension (pure_int8_ops.cpp)
- torch._int_mm dependency

With:
- CuPy RawKernels for custom CUDA
- CuPy array ops for everything else
- Zero build step, zero extension compilation

Works on your P40s (sm_61, DP4A supported) out of the box.
"""

import cupy as cp
import numpy as np
import math

# ============================================
# RAW CUDA KERNELS (compiled at import time)
# ============================================

_STOCHASTIC_SHIFT_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void stochastic_shift_kernel(
    const int* __restrict__ input,
    signed char* __restrict__ output,
    int shift_bits,
    unsigned int seed,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // LFSR per-element RNG
    unsigned int rng = seed ^ (unsigned int)idx;
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;

    int val = input[idx];

    if (shift_bits <= 0) {
        output[idx] = (signed char)max(-128, min(127, val));
        return;
    }

    int mask = (1 << shift_bits) - 1;
    int dropped = val & mask;
    int threshold = (int)(rng & (unsigned int)mask);

    int shifted = val >> shift_bits;

    if (dropped > threshold && val > 0) {
        shifted += 1;
    } else if ((-dropped) > threshold && val < 0) {
        shifted -= 1;
    }

    output[idx] = (signed char)max(-128, min(127, shifted));
}
''', 'stochastic_shift_kernel')


_INT8_ACTIVATION_LUT_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void int8_activation_lut_kernel(
    const signed char* __restrict__ input,
    signed char* __restrict__ output,
    const signed char* __restrict__ lut,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    output[idx] = lut[(int)input[idx] + 128];
}
''', 'int8_activation_lut_kernel')


_QUANTIZE_INPUT_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void quantize_input_kernel(
    const float* __restrict__ input,
    signed char* __restrict__ output,
    signed char* __restrict__ scale_shift,
    int M,
    int K
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // Find max absolute value in this row
    float max_val = 0.0f;
    for (int k = 0; k < K; k++) {
        float val = fabsf(input[row * K + k]);
        if (val > max_val) max_val = val;
    }

    // Compute power-of-2 shift
    int shift = 0;
    if (max_val > 127.0f) {
        shift = (int)ceilf(log2f(max_val / 127.0f));
    }
    shift = max(0, min(126, shift));

    float scale = (float)(1 << shift);

    // Quantize
    for (int k = 0; k < K; k++) {
        float val = input[row * K + k] / scale;
        int quantized = (int)roundf(val);
        output[row * K + k] = (signed char)max(-128, min(127, quantized));
    }

    scale_shift[row] = (signed char)shift;
}
''', 'quantize_input_kernel')


_DEQUANTIZE_OUTPUT_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void dequantize_output_kernel(
    const signed char* __restrict__ input,
    float* __restrict__ output,
    const signed char* __restrict__ input_scale_shift,
    const int* __restrict__ scale_multiplier,
    const signed char* __restrict__ scale_shift,
    int M,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    signed char val = input[row * N + col];
    int in_shift = (int)input_scale_shift[row];
    int mult = scale_multiplier[col];
    int out_shift = (int)scale_shift[col];

    float scale = (float)mult * powf(2.0f, (float)(in_shift - out_shift));
    output[row * N + col] = (float)val * scale;
}
''', 'dequantize_output_kernel')


# DP4A-based GEMM for SM_61+ (Pascal/P40)
_INT8_GEMM_KERNEL = cp.RawKernel(r'''
#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

__device__ __forceinline__ unsigned int xorshift32(unsigned int state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ __forceinline__ int dp4a(int a, int b, int acc) {
    #if __CUDA_ARCH__ >= 610
        return __dp4a(a, b, acc);
    #else
        signed char* ab = reinterpret_cast<signed char*>(&a);
        signed char* bb = reinterpret_cast<signed char*>(&b);
        acc += (int)ab[0] * bb[0];
        acc += (int)ab[1] * bb[1];
        acc += (int)ab[2] * bb[2];
        acc += (int)ab[3] * bb[3];
        return acc;
    #endif
}

__device__ __forceinline__ signed char stochastic_shift_to_int8(
    int value, int shift_bits, unsigned int rng
) {
    if (shift_bits <= 0) return (signed char)max(-128, min(127, value));
    int mask = (1 << shift_bits) - 1;
    int dropped = value & mask;
    unsigned int threshold = rng & (unsigned int)mask;
    int rounded = value >> shift_bits;
    if (dropped > (int)threshold && value > 0) rounded += 1;
    else if ((-dropped) > (int)threshold && value < 0) rounded -= 1;
    return (signed char)max(-128, min(127, rounded));
}

extern "C" __global__
void pure_int8_gemm_kernel(
    const signed char* __restrict__ A,
    const signed char* __restrict__ B,
    signed char* __restrict__ C,
    const int* __restrict__ scale_multiplier,
    const signed char* __restrict__ scale_shift,
    int output_shift,
    int M, int N, int K,
    unsigned int random_seed
) {
    __shared__ signed char As[TILE_M][TILE_K];
    __shared__ signed char Bs[TILE_N][TILE_K];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_M + ty;
    int col = blockIdx.x * TILE_N + tx;

    int acc = 0;
    unsigned int rng_state = random_seed ^ (unsigned int)(row * N + col);
    rng_state = xorshift32(rng_state);

    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_K + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0;

        int b_col = t * TILE_K + ty;
        Bs[tx][ty] = (col < N && b_col < K) ? B[col * K + b_col] : 0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; k += 4) {
            int a_packed = *reinterpret_cast<const int*>(&As[ty][k]);
            int b_packed = *reinterpret_cast<const int*>(&Bs[tx][k]);
            acc = dp4a(a_packed, b_packed, acc);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        int multiplier = scale_multiplier[col];
        long long scaled = (long long)acc * multiplier;

        int total_shift = (int)scale_shift[col] + output_shift;
        rng_state = xorshift32(rng_state);

        int value_to_shift;
        if (total_shift >= 32) {
            value_to_shift = (int)(scaled >> 32);
            total_shift -= 32;
        } else {
            value_to_shift = (int)scaled;
        }

        C[row * N + col] = stochastic_shift_to_int8(value_to_shift, total_shift, rng_state);
    }
}
''', 'pure_int8_gemm_kernel')


# Fused GEMM + activation LUT
_INT8_GEMM_FUSED_ACTIVATION_KERNEL = cp.RawKernel(r'''
#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

__device__ __forceinline__ unsigned int xorshift32(unsigned int state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ __forceinline__ int dp4a(int a, int b, int acc) {
    #if __CUDA_ARCH__ >= 610
        return __dp4a(a, b, acc);
    #else
        signed char* ab = reinterpret_cast<signed char*>(&a);
        signed char* bb = reinterpret_cast<signed char*>(&b);
        acc += (int)ab[0] * bb[0];
        acc += (int)ab[1] * bb[1];
        acc += (int)ab[2] * bb[2];
        acc += (int)ab[3] * bb[3];
        return acc;
    #endif
}

__device__ __forceinline__ signed char stochastic_shift_to_int8(
    int value, int shift_bits, unsigned int rng
) {
    if (shift_bits <= 0) return (signed char)max(-128, min(127, value));
    int mask = (1 << shift_bits) - 1;
    int dropped = value & mask;
    unsigned int threshold = rng & (unsigned int)mask;
    int rounded = value >> shift_bits;
    if (dropped > (int)threshold && value > 0) rounded += 1;
    else if ((-dropped) > (int)threshold && value < 0) rounded -= 1;
    return (signed char)max(-128, min(127, rounded));
}

extern "C" __global__
void pure_int8_gemm_fused_activation_kernel(
    const signed char* __restrict__ A,
    const signed char* __restrict__ B,
    signed char* __restrict__ C,
    const int* __restrict__ scale_multiplier,
    const signed char* __restrict__ scale_shift,
    const signed char* __restrict__ activation_lut,
    int output_shift,
    int M, int N, int K,
    unsigned int random_seed
) {
    __shared__ signed char As[TILE_M][TILE_K];
    __shared__ signed char Bs[TILE_N][TILE_K];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_M + ty;
    int col = blockIdx.x * TILE_N + tx;

    int acc = 0;
    unsigned int rng_state = random_seed ^ (unsigned int)(row * N + col);
    rng_state = xorshift32(rng_state);

    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_K + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0;

        int b_col = t * TILE_K + ty;
        Bs[tx][ty] = (col < N && b_col < K) ? B[col * K + b_col] : 0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; k += 4) {
            int a_packed = *reinterpret_cast<const int*>(&As[ty][k]);
            int b_packed = *reinterpret_cast<const int*>(&Bs[tx][k]);
            acc = dp4a(a_packed, b_packed, acc);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        int multiplier = scale_multiplier[col];
        long long scaled = (long long)acc * multiplier;

        int total_shift = (int)scale_shift[col] + output_shift;
        rng_state = xorshift32(rng_state);

        int value_to_shift;
        if (total_shift >= 32) {
            value_to_shift = (int)(scaled >> 32);
            total_shift -= 32;
        } else {
            value_to_shift = (int)scaled;
        }

        signed char pre_act = stochastic_shift_to_int8(value_to_shift, total_shift, rng_state);
        C[row * N + col] = activation_lut[(int)pre_act + 128];
    }
}
''', 'pure_int8_gemm_fused_activation_kernel')


# ============================================
# LUT GENERATION (CuPy arrays, GPU-resident)
# ============================================

def generate_silu_lut_gpu():
    """Generate SiLU LUT directly as CuPy array on GPU."""
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        x = float(i - 128)
        if x >= 0:
            silu = x / (1.0 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            silu = x * exp_x / (1.0 + exp_x)
        lut[i] = max(-128, min(127, int(round(silu))))
    return cp.asarray(lut)


def generate_gelu_lut_gpu():
    """Generate GELU LUT directly as CuPy array on GPU."""
    lut = np.zeros(256, dtype=np.int8)
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    for i in range(256):
        x = float(i - 128)
        inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
        gelu = 0.5 * x * (1.0 + math.tanh(inner))
        lut[i] = max(-128, min(127, int(round(gelu))))
    return cp.asarray(lut)


def generate_relu_lut_gpu():
    """Generate ReLU LUT directly as CuPy array on GPU."""
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        lut[i] = max(0, i - 128)
    return cp.asarray(lut)


# ============================================
# HIGH-LEVEL OPS (CuPy wrappers)
# ============================================

class CupyInt8Ops:
    """
    All INT8 operations via CuPy. No torch, no ctypes, no build step.

    Usage:
        ops = CupyInt8Ops()
        out = ops.gemm(a_int8, b_int8, scale_mult, scale_shift, output_shift)
        activated = ops.apply_activation(out, 'silu')
        fused = ops.gemm_fused(a, b, scale_mult, scale_shift, output_shift, 'silu')
    """

    def __init__(self):
        self.silu_lut = generate_silu_lut_gpu()
        self.gelu_lut = generate_gelu_lut_gpu()
        self.relu_lut = generate_relu_lut_gpu()
        self._rng_counter = 0

    def _next_seed(self):
        self._rng_counter += 1
        return np.uint32(hash(self._rng_counter) & 0xFFFFFFFF)

    def _get_lut(self, activation):
        luts = {'silu': self.silu_lut, 'gelu': self.gelu_lut, 'relu': self.relu_lut}
        if activation not in luts:
            raise ValueError(f"Unknown activation: {activation}. Use: {list(luts.keys())}")
        return luts[activation]

    # ------------------------------------------
    # Core GEMM: INT8 @ INT8 -> INT8
    # ------------------------------------------

    def gemm(self, A, B, scale_multiplier, scale_shift, output_shift=8):
        """
        Pure INT8 GEMM with stochastic rounding.

        Args:
            A: (M, K) int8 CuPy array
            B: (N, K) int8 CuPy array (transposed internally)
            scale_multiplier: (N,) int32 per-channel scale
            scale_shift: (N,) int8 per-channel shift
            output_shift: int, global shift for INT32->INT8

        Returns:
            C: (M, N) int8 CuPy array
        """
        A = cp.ascontiguousarray(A, dtype=cp.int8)
        B = cp.ascontiguousarray(B, dtype=cp.int8)
        scale_multiplier = cp.ascontiguousarray(scale_multiplier, dtype=cp.int32)
        scale_shift = cp.ascontiguousarray(scale_shift, dtype=cp.int8)

        M, K = A.shape
        N = B.shape[0]
        assert B.shape[1] == K

        C = cp.empty((M, N), dtype=cp.int8)

        block = (32, 32)
        grid = ((N + 31) // 32, (M + 31) // 32)

        _INT8_GEMM_KERNEL(
            grid, block,
            (A, B, C, scale_multiplier, scale_shift,
             np.int32(output_shift), np.int32(M), np.int32(N), np.int32(K),
             self._next_seed())
        )

        return C

    # ------------------------------------------
    # Fused GEMM + Activation
    # ------------------------------------------

    def gemm_fused(self, A, B, scale_multiplier, scale_shift, output_shift=8,
                   activation='silu'):
        """
        Fused INT8 GEMM + LUT activation in one kernel launch.
        Zero intermediate memory traffic.

        Args:
            A: (M, K) int8
            B: (N, K) int8
            scale_multiplier: (N,) int32
            scale_shift: (N,) int8
            output_shift: int
            activation: 'silu', 'gelu', or 'relu'

        Returns:
            C: (M, N) int8 (post-activation)
        """
        A = cp.ascontiguousarray(A, dtype=cp.int8)
        B = cp.ascontiguousarray(B, dtype=cp.int8)
        scale_multiplier = cp.ascontiguousarray(scale_multiplier, dtype=cp.int32)
        scale_shift = cp.ascontiguousarray(scale_shift, dtype=cp.int8)
        lut = self._get_lut(activation)

        M, K = A.shape
        N = B.shape[0]
        C = cp.empty((M, N), dtype=cp.int8)

        block = (32, 32)
        grid = ((N + 31) // 32, (M + 31) // 32)

        _INT8_GEMM_FUSED_ACTIVATION_KERNEL(
            grid, block,
            (A, B, C, scale_multiplier, scale_shift, lut,
             np.int32(output_shift), np.int32(M), np.int32(N), np.int32(K),
             self._next_seed())
        )

        return C

    # ------------------------------------------
    # Standalone Activation (LUT)
    # ------------------------------------------

    def apply_activation(self, x, activation='silu'):
        """
        Apply activation via 256-byte LUT. O(1) per element.

        Args:
            x: int8 CuPy array (any shape)
            activation: 'silu', 'gelu', or 'relu'

        Returns:
            int8 CuPy array (same shape)
        """
        x = cp.ascontiguousarray(x, dtype=cp.int8)
        lut = self._get_lut(activation)

        output = cp.empty_like(x)
        N = x.size

        threads = 256
        blocks = (N + threads - 1) // threads

        _INT8_ACTIVATION_LUT_KERNEL(
            (blocks,), (threads,),
            (x.ravel(), output.ravel(), lut, np.int32(N))
        )

        return output.reshape(x.shape)

    def apply_activation_custom(self, x, lut):
        """Apply custom 256-byte LUT activation."""
        x = cp.ascontiguousarray(x, dtype=cp.int8)
        lut = cp.ascontiguousarray(lut, dtype=cp.int8)
        assert lut.size == 256

        output = cp.empty_like(x)
        N = x.size
        threads = 256
        blocks = (N + threads - 1) // threads

        _INT8_ACTIVATION_LUT_KERNEL(
            (blocks,), (threads,),
            (x.ravel(), output.ravel(), lut, np.int32(N))
        )
        return output.reshape(x.shape)

    # ------------------------------------------
    # Quantization / Dequantization
    # ------------------------------------------

    def quantize_input(self, x_float):
        """
        Float -> INT8 with power-of-2 scaling.

        Args:
            x_float: (M, K) float32 CuPy array

        Returns:
            x_int8: (M, K) int8
            scale_shift: (M,) int8
        """
        x_float = cp.ascontiguousarray(x_float, dtype=cp.float32)
        M, K = x_float.shape

        x_int8 = cp.empty((M, K), dtype=cp.int8)
        scale_shift = cp.empty(M, dtype=cp.int8)

        threads = 256
        blocks = (M + threads - 1) // threads

        _QUANTIZE_INPUT_KERNEL(
            (blocks,), (threads,),
            (x_float, x_int8, scale_shift, np.int32(M), np.int32(K))
        )

        return x_int8, scale_shift

    def dequantize_output(self, x_int8, input_scale_shift, scale_multiplier, scale_shift):
        """
        INT8 -> Float for final logits layer.

        Args:
            x_int8: (M, N) int8
            input_scale_shift: (M,) int8
            scale_multiplier: (N,) int32
            scale_shift: (N,) int8

        Returns:
            output: (M, N) float32
        """
        x_int8 = cp.ascontiguousarray(x_int8, dtype=cp.int8)
        input_scale_shift = cp.ascontiguousarray(input_scale_shift, dtype=cp.int8)
        scale_multiplier = cp.ascontiguousarray(scale_multiplier, dtype=cp.int32)
        scale_shift = cp.ascontiguousarray(scale_shift, dtype=cp.int8)

        M, N = x_int8.shape
        output = cp.empty((M, N), dtype=cp.float32)

        block = (16, 16)
        grid = ((N + 15) // 16, (M + 15) // 16)

        _DEQUANTIZE_OUTPUT_KERNEL(
            grid, block,
            (x_int8, output, input_scale_shift, scale_multiplier, scale_shift,
             np.int32(M), np.int32(N))
        )

        return output

    # ------------------------------------------
    # Stochastic Rounding (standalone)
    # ------------------------------------------

    def stochastic_shift(self, values_int32, shift_bits):
        """
        INT32 -> INT8 with stochastic rounding on GPU.

        Args:
            values_int32: int32 CuPy array
            shift_bits: int

        Returns:
            int8 CuPy array
        """
        values_int32 = cp.ascontiguousarray(values_int32, dtype=cp.int32)
        output = cp.empty(values_int32.shape, dtype=cp.int8)
        N = values_int32.size

        threads = 256
        blocks = (N + threads - 1) // threads

        _STOCHASTIC_SHIFT_KERNEL(
            (blocks,), (threads,),
            (values_int32.ravel(), output.ravel(),
             np.int32(shift_bits), self._next_seed(), np.int32(N))
        )

        return output.reshape(values_int32.shape)

    # ------------------------------------------
    # CuPy matmul fallback (no custom kernel)
    # ------------------------------------------

    def matmul_int8_simple(self, A_int8, B_int8):
        """
        Simple INT8 matmul via CuPy (INT32 accumulation).
        No stochastic rounding, no scaling — just A @ B.T in int32.

        For when you need the raw accumulator.
        """
        return cp.matmul(A_int8.astype(cp.int32), B_int8.astype(cp.int32).T)


# ============================================
# CONVENIENCE: numpy-compatible wrappers
# ============================================

def to_gpu(arr):
    """numpy/list -> CuPy array on GPU."""
    if isinstance(arr, cp.ndarray):
        return arr
    return cp.asarray(arr)


def to_cpu(arr):
    """CuPy array -> numpy on CPU."""
    if isinstance(arr, np.ndarray):
        return arr
    return cp.asnumpy(arr)
