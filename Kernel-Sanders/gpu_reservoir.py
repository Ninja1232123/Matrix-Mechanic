#!/usr/bin/env python3
"""
GPU Reservoir — CuPy RawKernel (P40 SM_61)
============================================

Fused batched CUDA kernel for the dual-state Julia set reservoir.
Replaces the CPU NumPy per-token loop with one kernel launch per token
processing the entire batch in parallel.

Grid: (batch_size,)   Block: (cdim,)
Each thread = one complex dimension of one sequence in the batch.

All state lives on GPU (CuPy int32 arrays). No per-token CPU<->GPU transfer.
"""

import cupy as cp
import numpy as np
import re as _re

# ===========================================================================
# FOLD MODE: Hard bitmask (original) vs Soft softsign (smooth, no discontinuity)
# ===========================================================================

# Hard fold: bitmask wrap at ±2.0 then π/2 scale. Creates LE≈+2.08 at dim=512.
_FOLD_HARD_DEF = r'''
#define FOLD(re, im) do { \
    re = ((re + 131072) & 0x3FFFF) - 131072; \
    re = (int)(((long long)re * 102944LL + 32768LL) >> 16); \
    im = ((im + 131072) & 0x3FFFF) - 131072; \
    im = (int)(((long long)im * 102944LL + 32768LL) >> 16); \
} while(0)
'''

# Soft fold: softsign z_out = FSCALE * z / (FSCALE + |z|). Smooth, bounded.
# FSCALE controls amplification: 65536=1.0(neutral), 82000=1.25(mild), 102944=pi/2(strong).
# f'(0) = 1.0 regardless of FSCALE. f(∞) = FSCALE (saturation level).
# FSCALE is set by _compile_soft_fold_def() based on fold_scale parameter.
def _compile_soft_fold_def(fold_scale_fp=65536):
    """Generate soft fold macro with configurable scale. fold_scale_fp is in fp16."""
    return (
        f'#define FOLD(re, im) do {{ \\\n'
        f'    int _ar = re < 0 ? -re : re; \\\n'
        f'    int _ai = im < 0 ? -im : im; \\\n'
        f'    re = (int)({fold_scale_fp}LL * (long long)re / ({fold_scale_fp}LL + (long long)_ar)); \\\n'
        f'    im = (int)({fold_scale_fp}LL * (long long)im / ({fold_scale_fp}LL + (long long)_ai)); \\\n'
        f'}} while(0)\n'
    )


def _inject_fold_macro(kernel_src):
    """Replace inline 4-line hard fold blocks with FOLD(re, im) macro calls."""
    pattern = (
        r'( +)(\w+) = \(\(\2 \+ 131072\) & 0x3FFFF\) - 131072;\n'
        r' +\2 = \(int\)\(\(\(long long\)\2 \* 102944LL \+ 32768LL\) >> 16\);\n'
        r' +(\w+) = \(\(\3 \+ 131072\) & 0x3FFFF\) - 131072;\n'
        r' +\3 = \(int\)\(\(\(long long\)\3 \* 102944LL \+ 32768LL\) >> 16\);'
    )
    def repl(m):
        return f'{m.group(1)}FOLD({m.group(2)}, {m.group(3)});'
    return _re.sub(pattern, repl, kernel_src)


_compiled_kernels = {}

def _get_kernels(soft_fold=False, fold_scale=1.0):
    """Compile and cache CUDA kernels with the chosen fold mode.

    fold_scale: for soft fold, controls saturation (1.0=neutral, 1.25=mild amplify, pi/2=strong).
    """
    fold_scale_fp = int(round(fold_scale * 65536))
    key = f'soft_{fold_scale_fp}' if soft_fold else 'hard'
    if key in _compiled_kernels:
        return _compiled_kernels[key]

    fold_def = _compile_soft_fold_def(fold_scale_fp) if soft_fold else _FOLD_HARD_DEF
    kernels = {
        'step': cp.RawKernel(
            fold_def + _inject_fold_macro(_RESERVOIR_KERNEL_SRC), 'reservoir_step'),
        'fused': cp.RawKernel(
            fold_def + _inject_fold_macro(_FUSED_SEQ_KERNEL_SRC), 'reservoir_seq'),
        'adaptive': cp.RawKernel(
            fold_def + _inject_fold_macro(_ADAPTIVE_C_KERNEL_SRC), 'reservoir_seq_adaptive'),
        'energy': cp.RawKernel(
            fold_def + _inject_fold_macro(_ENERGY_C_KERNEL_SRC), 'reservoir_seq_energy'),
        'pertoken': cp.RawKernel(
            fold_def + _inject_fold_macro(_PERTOKEN_C_KERNEL_SRC), 'reservoir_seq_pertoken_c'),
    }
    if soft_fold:
        fold_name = f"SOFT (softsign, scale={fold_scale:.2f})"
    else:
        fold_name = "HARD (bitmask)"
    print(f"  [GPU] Compiled kernels with {fold_name} fold")
    _compiled_kernels[key] = kernels
    return kernels


# ===========================================================================
# CUDA KERNEL SOURCE
# ===========================================================================

_RESERVOIR_KERNEL_SRC = r'''
extern "C" __global__
void reservoir_step(
    // --- State arrays (batch_size, cdim) int32, read+write ---
    int* __restrict__ mem_re,
    int* __restrict__ mem_im,
    int* __restrict__ sense_re,
    int* __restrict__ sense_im,

    // --- Injection for this token (batch_size, cdim) int32, read-only ---
    const int* __restrict__ inj_re,
    const int* __restrict__ inj_im,

    // --- C values: sense (n_layers, cdim) int32 ---
    const int* __restrict__ c_sense_re,
    const int* __restrict__ c_sense_im,

    // --- C values: memory (n_layers, cdim) int32 ---
    const int* __restrict__ c_mem_re,
    const int* __restrict__ c_mem_im,

    // --- Output: interference (batch_size, cdim) int32 ---
    int* __restrict__ out_re,
    int* __restrict__ out_im,

    // --- Layer outputs: sense per-layer (batch_size, n_layers, cdim) int32 ---
    int* __restrict__ layer_out_re,
    int* __restrict__ layer_out_im,

    // --- Fixed-point mixing constants (int32, broadcast) ---
    int decay_fp,       // sense_decay in fixed-point
    int mem_a_fp,       // mem_alpha in fixed-point
    int mem_b_fp,       // 1 - mem_alpha in fixed-point
    int sense_a_fp,     // sense_alpha in fixed-point
    int sense_b_fp,     // 1 - sense_alpha in fixed-point
    int res_fp,         // residual_strength in fixed-point
    int res_inv_fp,     // 1 - residual_strength in fixed-point

    // --- Dimensions ---
    int cdim,
    int n_layers,
    int mem_iter,
    int sense_iter,
    int batch_size
) {
    int bid = blockIdx.x;   // batch index
    int tid = blockIdx.y * blockDim.x + threadIdx.x;  // complex dimension index

    if (bid >= batch_size || tid >= cdim) return;

    int idx = bid * cdim + tid;  // flat index into (batch, cdim)

    // Load state
    int mr = mem_re[idx];
    int mi = mem_im[idx];
    int sr = sense_re[idx];
    int si = sense_im[idx];

    // Load injection
    int ir = inj_re[idx];
    int ii = inj_im[idx];

    // ==========================================
    // 1. SENSE DECAY
    // ==========================================
    sr = (int)(((long long)decay_fp * (long long)sr + 32768LL) >> 16);
    si = (int)(((long long)decay_fp * (long long)si + 32768LL) >> 16);

    // ==========================================
    // 2. MIX INJECTION INTO MEMORY (slow)
    // ==========================================
    mr = (int)(((long long)mem_a_fp * (long long)ir + 32768LL) >> 16)
       + (int)(((long long)mem_b_fp * (long long)mr + 32768LL) >> 16);
    mi = (int)(((long long)mem_a_fp * (long long)ii + 32768LL) >> 16)
       + (int)(((long long)mem_b_fp * (long long)mi + 32768LL) >> 16);

    // ==========================================
    // 3. MIX INJECTION INTO SENSE (fast)
    // ==========================================
    sr = (int)(((long long)sense_a_fp * (long long)ir + 32768LL) >> 16)
       + (int)(((long long)sense_b_fp * (long long)sr + 32768LL) >> 16);
    si = (int)(((long long)sense_a_fp * (long long)ii + 32768LL) >> 16)
       + (int)(((long long)sense_b_fp * (long long)si + 32768LL) >> 16);

    // ==========================================
    // 4. JULIA ITERATIONS - MEMORY (with residual)
    // ==========================================
    int pre_mr = mr;
    int pre_mi = mi;

    for (int li = 0; li < n_layers; li++) {
        int c_offset = li * cdim + tid;
        int cr = c_mem_re[c_offset];
        int ci = c_mem_im[c_offset];

        for (int it = 0; it < mem_iter; it++) {
            // Torus fold
            mr = ((mr + 131072) & 0x3FFFF) - 131072;
            mr = (int)(((long long)mr * 102944LL + 32768LL) >> 16);
            mi = ((mi + 131072) & 0x3FFFF) - 131072;
            mi = (int)(((long long)mi * 102944LL + 32768LL) >> 16);
            // z^2 + c
            long long zr64 = (long long)mr;
            long long zi64 = (long long)mi;
            int re2  = (int)((zr64 * zr64 + 32768LL) >> 16);
            int im2  = (int)((zi64 * zi64 + 32768LL) >> 16);
            int reim = (int)((zr64 * zi64 + 32768LL) >> 16);
            mr = re2 - im2 + cr;
            mi = (reim << 1) + ci;
        }
    }

    // Residual connection
    mr = (int)(((long long)res_inv_fp * (long long)mr + 32768LL) >> 16)
       + (int)(((long long)res_fp * (long long)pre_mr + 32768LL) >> 16);
    mi = (int)(((long long)res_inv_fp * (long long)mi + 32768LL) >> 16)
       + (int)(((long long)res_fp * (long long)pre_mi + 32768LL) >> 16);

    // ==========================================
    // 5. JULIA ITERATIONS - SENSE (capture layer outputs)
    // ==========================================
    for (int li = 0; li < n_layers; li++) {
        int c_offset = li * cdim + tid;
        int cr = c_sense_re[c_offset];
        int ci = c_sense_im[c_offset];

        for (int it = 0; it < sense_iter; it++) {
            // Torus fold
            sr = ((sr + 131072) & 0x3FFFF) - 131072;
            sr = (int)(((long long)sr * 102944LL + 32768LL) >> 16);
            si = ((si + 131072) & 0x3FFFF) - 131072;
            si = (int)(((long long)si * 102944LL + 32768LL) >> 16);
            // z^2 + c
            long long zr64 = (long long)sr;
            long long zi64 = (long long)si;
            int re2  = (int)((zr64 * zr64 + 32768LL) >> 16);
            int im2  = (int)((zi64 * zi64 + 32768LL) >> 16);
            int reim = (int)((zr64 * zi64 + 32768LL) >> 16);
            sr = re2 - im2 + cr;
            si = (reim << 1) + ci;
        }

        // Capture per-layer output
        int lo_idx = bid * n_layers * cdim + li * cdim + tid;
        layer_out_re[lo_idx] = sr;
        layer_out_im[lo_idx] = si;
    }

    // ==========================================
    // 6. OUTPUT = MEMORY x SENSE (complex multiply)
    // ==========================================
    long long ar64 = (long long)mr;
    long long ai64 = (long long)mi;
    long long br64 = (long long)sr;
    long long bi64 = (long long)si;
    int o_re = (int)((ar64 * br64 + 32768LL) >> 16) - (int)((ai64 * bi64 + 32768LL) >> 16);
    int o_im = (int)((ar64 * bi64 + 32768LL) >> 16) + (int)((ai64 * br64 + 32768LL) >> 16);

    // Torus fold output
    o_re = ((o_re + 131072) & 0x3FFFF) - 131072;
    o_re = (int)(((long long)o_re * 102944LL + 32768LL) >> 16);
    o_im = ((o_im + 131072) & 0x3FFFF) - 131072;
    o_im = (int)(((long long)o_im * 102944LL + 32768LL) >> 16);

    // ==========================================
    // 7. WRITE BACK
    // ==========================================
    mem_re[idx] = mr;
    mem_im[idx] = mi;
    sense_re[idx] = sr;
    sense_im[idx] = si;
    out_re[idx] = o_re;
    out_im[idx] = o_im;
}
'''

# Lazy-compiled per fold mode — see _get_kernels()

# ===========================================================================
# FUSED FULL-SEQUENCE KERNEL (one launch for entire sequence)
# ===========================================================================
_FUSED_SEQ_KERNEL_SRC = r'''
extern "C" __global__
void reservoir_seq(
    // --- State arrays (batch_size, cdim) int32, read+write ---
    int* __restrict__ mem_re,
    int* __restrict__ mem_im,
    int* __restrict__ sense_re,
    int* __restrict__ sense_im,

    // --- Injection sequence (batch_size, seq_len, cdim) int32, contiguous ---
    const int* __restrict__ inj_re_seq,
    const int* __restrict__ inj_im_seq,

    // --- C values: sense (n_layers, cdim) int32 ---
    const int* __restrict__ c_sense_re,
    const int* __restrict__ c_sense_im,

    // --- C values: memory (n_layers, cdim) int32 ---
    const int* __restrict__ c_mem_re,
    const int* __restrict__ c_mem_im,

    // --- Output sequence (batch_size, seq_len, cdim) int32 ---
    int* __restrict__ out_re_seq,
    int* __restrict__ out_im_seq,

    // --- Layer outputs (batch_size, seq_len, n_layers, cdim) int32 ---
    int* __restrict__ layer_re_seq,
    int* __restrict__ layer_im_seq,

    // --- Fixed-point mixing constants ---
    int decay_fp,
    int mem_a_fp, int mem_b_fp,
    int sense_a_fp, int sense_b_fp,
    int res_fp, int res_inv_fp,

    // --- Dimensions ---
    int cdim, int n_layers, int mem_iter, int sense_iter,
    int batch_size, int seq_len
) {
    int bid = blockIdx.x;
    int tid = blockIdx.y * blockDim.x + threadIdx.x;
    if (bid >= batch_size || tid >= cdim) return;

    int state_idx = bid * cdim + tid;

    // Load state into registers
    int mr = mem_re[state_idx];
    int mi = mem_im[state_idx];
    int sr = sense_re[state_idx];
    int si = sense_im[state_idx];

    // Loop over all tokens in sequence
    for (int t = 0; t < seq_len; t++) {
        int inj_idx = (bid * seq_len + t) * cdim + tid;
        int ir = inj_re_seq[inj_idx];
        int ii = inj_im_seq[inj_idx];

        // 1. SENSE DECAY
        sr = (int)(((long long)decay_fp * (long long)sr + 32768LL) >> 16);
        si = (int)(((long long)decay_fp * (long long)si + 32768LL) >> 16);

        // 2. MIX INJECTION INTO MEMORY
        mr = (int)(((long long)mem_a_fp * (long long)ir + 32768LL) >> 16)
           + (int)(((long long)mem_b_fp * (long long)mr + 32768LL) >> 16);
        mi = (int)(((long long)mem_a_fp * (long long)ii + 32768LL) >> 16)
           + (int)(((long long)mem_b_fp * (long long)mi + 32768LL) >> 16);

        // 3. MIX INJECTION INTO SENSE
        sr = (int)(((long long)sense_a_fp * (long long)ir + 32768LL) >> 16)
           + (int)(((long long)sense_b_fp * (long long)sr + 32768LL) >> 16);
        si = (int)(((long long)sense_a_fp * (long long)ii + 32768LL) >> 16)
           + (int)(((long long)sense_b_fp * (long long)si + 32768LL) >> 16);

        // 4. JULIA ITERATIONS - MEMORY (with residual)
        int pre_mr = mr, pre_mi = mi;
        for (int li = 0; li < n_layers; li++) {
            int c_offset = li * cdim + tid;
            int cr = c_mem_re[c_offset];
            int ci = c_mem_im[c_offset];
            for (int it = 0; it < mem_iter; it++) {
                mr = ((mr + 131072) & 0x3FFFF) - 131072;
                mr = (int)(((long long)mr * 102944LL + 32768LL) >> 16);
                mi = ((mi + 131072) & 0x3FFFF) - 131072;
                mi = (int)(((long long)mi * 102944LL + 32768LL) >> 16);
                long long zr64 = (long long)mr;
                long long zi64 = (long long)mi;
                int re2  = (int)((zr64 * zr64 + 32768LL) >> 16);
                int im2  = (int)((zi64 * zi64 + 32768LL) >> 16);
                int reim = (int)((zr64 * zi64 + 32768LL) >> 16);
                mr = re2 - im2 + cr;
                mi = (reim << 1) + ci;
            }
        }
        mr = (int)(((long long)res_inv_fp * (long long)mr + 32768LL) >> 16)
           + (int)(((long long)res_fp * (long long)pre_mr + 32768LL) >> 16);
        mi = (int)(((long long)res_inv_fp * (long long)mi + 32768LL) >> 16)
           + (int)(((long long)res_fp * (long long)pre_mi + 32768LL) >> 16);

        // 5. JULIA ITERATIONS - SENSE (capture per-layer)
        for (int li = 0; li < n_layers; li++) {
            int c_offset = li * cdim + tid;
            int cr = c_sense_re[c_offset];
            int ci = c_sense_im[c_offset];
            for (int it = 0; it < sense_iter; it++) {
                sr = ((sr + 131072) & 0x3FFFF) - 131072;
                sr = (int)(((long long)sr * 102944LL + 32768LL) >> 16);
                si = ((si + 131072) & 0x3FFFF) - 131072;
                si = (int)(((long long)si * 102944LL + 32768LL) >> 16);
                long long zr64 = (long long)sr;
                long long zi64 = (long long)si;
                int re2  = (int)((zr64 * zr64 + 32768LL) >> 16);
                int im2  = (int)((zi64 * zi64 + 32768LL) >> 16);
                int reim = (int)((zr64 * zi64 + 32768LL) >> 16);
                sr = re2 - im2 + cr;
                si = (reim << 1) + ci;
            }
            // Store per-layer sense output
            int lo_idx = ((bid * seq_len + t) * n_layers + li) * cdim + tid;
            layer_re_seq[lo_idx] = sr;
            layer_im_seq[lo_idx] = si;
        }

        // 6. OUTPUT = MEMORY x SENSE (complex multiply + torus fold)
        long long ar64 = (long long)mr, ai64 = (long long)mi;
        long long br64 = (long long)sr, bi64 = (long long)si;
        int o_re = (int)((ar64 * br64 + 32768LL) >> 16) - (int)((ai64 * bi64 + 32768LL) >> 16);
        int o_im = (int)((ar64 * bi64 + 32768LL) >> 16) + (int)((ai64 * br64 + 32768LL) >> 16);
        o_re = ((o_re + 131072) & 0x3FFFF) - 131072;
        o_re = (int)(((long long)o_re * 102944LL + 32768LL) >> 16);
        o_im = ((o_im + 131072) & 0x3FFFF) - 131072;
        o_im = (int)(((long long)o_im * 102944LL + 32768LL) >> 16);

        // Store output
        int out_idx = (bid * seq_len + t) * cdim + tid;
        out_re_seq[out_idx] = o_re;
        out_im_seq[out_idx] = o_im;
    }

    // Write back final state
    mem_re[state_idx] = mr;
    mem_im[state_idx] = mi;
    sense_re[state_idx] = sr;
    sense_im[state_idx] = si;
}
'''


# ===========================================================================
# ADAPTIVE C-VALUE FUSED KERNEL (landscape evolves per-token)
# ===========================================================================
_ADAPTIVE_C_KERNEL_SRC = r'''
extern "C" __global__
void reservoir_seq_adaptive(
    // --- State arrays (batch_size, cdim) int32, read+write ---
    int* __restrict__ mem_re,
    int* __restrict__ mem_im,
    int* __restrict__ sense_re,
    int* __restrict__ sense_im,

    // --- Injection sequence (batch_size, seq_len, cdim) int32, contiguous ---
    const int* __restrict__ inj_re_seq,
    const int* __restrict__ inj_im_seq,

    // --- PER-BATCH c values (batch_size, n_layers, cdim) int32 - WRITABLE, evolving ---
    int* __restrict__ c_sense_re_batch,
    int* __restrict__ c_sense_im_batch,
    int* __restrict__ c_mem_re_batch,
    int* __restrict__ c_mem_im_batch,

    // --- Output sequence (batch_size, seq_len, cdim) int32 ---
    int* __restrict__ out_re_seq,
    int* __restrict__ out_im_seq,

    // --- Layer outputs (batch_size, seq_len, n_layers, cdim) int32 ---
    int* __restrict__ layer_re_seq,
    int* __restrict__ layer_im_seq,

    // --- Fixed-point mixing constants ---
    int decay_fp,
    int mem_a_fp, int mem_b_fp,
    int sense_a_fp, int sense_b_fp,
    int res_fp, int res_inv_fp,

    // --- Adaptive feedback scale (fixed-point) ---
    int adapt_scale,

    // --- Dimensions ---
    int cdim, int n_layers, int mem_iter, int sense_iter,
    int batch_size, int seq_len
) {
    int bid = blockIdx.x;
    int tid = blockIdx.y * blockDim.x + threadIdx.x;
    if (bid >= batch_size || tid >= cdim) return;

    int state_idx = bid * cdim + tid;

    // Load state into registers
    int mr = mem_re[state_idx];
    int mi = mem_im[state_idx];
    int sr = sense_re[state_idx];
    int si = sense_im[state_idx];

    // Loop over all tokens in sequence
    for (int t = 0; t < seq_len; t++) {
        int inj_idx = (bid * seq_len + t) * cdim + tid;
        int ir = inj_re_seq[inj_idx];
        int ii = inj_im_seq[inj_idx];

        // 1. SENSE DECAY
        sr = (int)(((long long)decay_fp * (long long)sr + 32768LL) >> 16);
        si = (int)(((long long)decay_fp * (long long)si + 32768LL) >> 16);

        // 2. MIX INJECTION INTO MEMORY
        mr = (int)(((long long)mem_a_fp * (long long)ir + 32768LL) >> 16)
           + (int)(((long long)mem_b_fp * (long long)mr + 32768LL) >> 16);
        mi = (int)(((long long)mem_a_fp * (long long)ii + 32768LL) >> 16)
           + (int)(((long long)mem_b_fp * (long long)mi + 32768LL) >> 16);

        // 3. MIX INJECTION INTO SENSE
        sr = (int)(((long long)sense_a_fp * (long long)ir + 32768LL) >> 16)
           + (int)(((long long)sense_b_fp * (long long)sr + 32768LL) >> 16);
        si = (int)(((long long)sense_a_fp * (long long)ii + 32768LL) >> 16)
           + (int)(((long long)sense_b_fp * (long long)si + 32768LL) >> 16);

        // 4. JULIA ITERATIONS - MEMORY (with residual)
        int pre_mr = mr, pre_mi = mi;
        for (int li = 0; li < n_layers; li++) {
            int bc_offset = (bid * n_layers + li) * cdim + tid;
            int cr = c_mem_re_batch[bc_offset];
            int ci = c_mem_im_batch[bc_offset];
            for (int it = 0; it < mem_iter; it++) {
                mr = ((mr + 131072) & 0x3FFFF) - 131072;
                mr = (int)(((long long)mr * 102944LL + 32768LL) >> 16);
                mi = ((mi + 131072) & 0x3FFFF) - 131072;
                mi = (int)(((long long)mi * 102944LL + 32768LL) >> 16);
                long long zr64 = (long long)mr;
                long long zi64 = (long long)mi;
                int re2  = (int)((zr64 * zr64 + 32768LL) >> 16);
                int im2  = (int)((zi64 * zi64 + 32768LL) >> 16);
                int reim = (int)((zr64 * zi64 + 32768LL) >> 16);
                mr = re2 - im2 + cr;
                mi = (reim << 1) + ci;
            }
        }
        mr = (int)(((long long)res_inv_fp * (long long)mr + 32768LL) >> 16)
           + (int)(((long long)res_fp * (long long)pre_mr + 32768LL) >> 16);
        mi = (int)(((long long)res_inv_fp * (long long)mi + 32768LL) >> 16)
           + (int)(((long long)res_fp * (long long)pre_mi + 32768LL) >> 16);

        // 5. JULIA ITERATIONS - SENSE (capture per-layer)
        for (int li = 0; li < n_layers; li++) {
            int bc_offset = (bid * n_layers + li) * cdim + tid;
            int cr = c_sense_re_batch[bc_offset];
            int ci = c_sense_im_batch[bc_offset];
            for (int it = 0; it < sense_iter; it++) {
                sr = ((sr + 131072) & 0x3FFFF) - 131072;
                sr = (int)(((long long)sr * 102944LL + 32768LL) >> 16);
                si = ((si + 131072) & 0x3FFFF) - 131072;
                si = (int)(((long long)si * 102944LL + 32768LL) >> 16);
                long long zr64 = (long long)sr;
                long long zi64 = (long long)si;
                int re2  = (int)((zr64 * zr64 + 32768LL) >> 16);
                int im2  = (int)((zi64 * zi64 + 32768LL) >> 16);
                int reim = (int)((zr64 * zi64 + 32768LL) >> 16);
                sr = re2 - im2 + cr;
                si = (reim << 1) + ci;
            }
            int lo_idx = ((bid * seq_len + t) * n_layers + li) * cdim + tid;
            layer_re_seq[lo_idx] = sr;
            layer_im_seq[lo_idx] = si;
        }

        // 6. OUTPUT = MEMORY x SENSE (complex multiply + torus fold)
        long long ar64 = (long long)mr, ai64 = (long long)mi;
        long long br64 = (long long)sr, bi64 = (long long)si;
        int o_re = (int)((ar64 * br64 + 32768LL) >> 16) - (int)((ai64 * bi64 + 32768LL) >> 16);
        int o_im = (int)((ar64 * bi64 + 32768LL) >> 16) + (int)((ai64 * br64 + 32768LL) >> 16);
        o_re = ((o_re + 131072) & 0x3FFFF) - 131072;
        o_re = (int)(((long long)o_re * 102944LL + 32768LL) >> 16);
        o_im = ((o_im + 131072) & 0x3FFFF) - 131072;
        o_im = (int)(((long long)o_im * 102944LL + 32768LL) >> 16);

        // Store output
        int out_idx = (bid * seq_len + t) * cdim + tid;
        out_re_seq[out_idx] = o_re;
        out_im_seq[out_idx] = o_im;

        // 7. ADAPTIVE FEEDBACK: output sculpts landscape for next token
        // c_delta = adapt_scale * output (element-wise, per dimension)
        // This lets the attractor geometry evolve with the sequence
        if (adapt_scale != 0) {
            int delta_re = (int)(((long long)adapt_scale * (long long)o_re + 32768LL) >> 16);
            int delta_im = (int)(((long long)adapt_scale * (long long)o_im + 32768LL) >> 16);
            for (int li = 0; li < n_layers; li++) {
                int bc_offset = (bid * n_layers + li) * cdim + tid;
                c_sense_re_batch[bc_offset] += delta_re;
                c_sense_im_batch[bc_offset] += delta_im;
                c_mem_re_batch[bc_offset] += delta_re;
                c_mem_im_batch[bc_offset] += delta_im;
            }
        }
    }

    // Write back final state
    mem_re[state_idx] = mr;
    mem_im[state_idx] = mi;
    sense_re[state_idx] = sr;
    sense_im[state_idx] = si;
}
'''


# ===========================================================================
# ENERGY-TO-C FEEDBACK KERNEL  (proportional, self-correcting)
# c_t = c_base + η * (z_sense - z_mem)  — anchored to base, never drifts
# ===========================================================================
_ENERGY_C_KERNEL_SRC = r'''
extern "C" __global__
void reservoir_seq_energy(
    // --- State arrays (batch_size, cdim) int32, read+write ---
    int* __restrict__ mem_re,
    int* __restrict__ mem_im,
    int* __restrict__ sense_re,
    int* __restrict__ sense_im,

    // --- Injection sequence (batch_size, seq_len, cdim) int32, contiguous ---
    const int* __restrict__ inj_re_seq,
    const int* __restrict__ inj_im_seq,

    // --- BASE c values (n_layers, cdim) int32 - READ-ONLY, shared across batch ---
    const int* __restrict__ c_sense_re_base,
    const int* __restrict__ c_sense_im_base,
    const int* __restrict__ c_mem_re_base,
    const int* __restrict__ c_mem_im_base,

    // --- Output sequence (batch_size, seq_len, cdim) int32 ---
    int* __restrict__ out_re_seq,
    int* __restrict__ out_im_seq,

    // --- Layer outputs (batch_size, seq_len, n_layers, cdim) int32 ---
    int* __restrict__ layer_re_seq,
    int* __restrict__ layer_im_seq,

    // --- Fixed-point mixing constants ---
    int decay_fp,
    int mem_a_fp, int mem_b_fp,
    int sense_a_fp, int sense_b_fp,
    int res_fp, int res_inv_fp,

    // --- Energy feedback scale (fixed-point) ---
    int energy_scale,

    // --- Dimensions ---
    int cdim, int n_layers, int mem_iter, int sense_iter,
    int batch_size, int seq_len
) {
    int bid = blockIdx.x;
    int tid = blockIdx.y * blockDim.x + threadIdx.x;
    if (bid >= batch_size || tid >= cdim) return;

    int state_idx = bid * cdim + tid;

    // Load state into registers
    int mr = mem_re[state_idx];
    int mi = mem_im[state_idx];
    int sr = sense_re[state_idx];
    int si = sense_im[state_idx];

    // Loop over all tokens in sequence
    for (int t = 0; t < seq_len; t++) {
        int inj_idx = (bid * seq_len + t) * cdim + tid;
        int ir = inj_re_seq[inj_idx];
        int ii = inj_im_seq[inj_idx];

        // 1. SENSE DECAY
        sr = (int)(((long long)decay_fp * (long long)sr + 32768LL) >> 16);
        si = (int)(((long long)decay_fp * (long long)si + 32768LL) >> 16);

        // 2. MIX INJECTION INTO MEMORY
        mr = (int)(((long long)mem_a_fp * (long long)ir + 32768LL) >> 16)
           + (int)(((long long)mem_b_fp * (long long)mr + 32768LL) >> 16);
        mi = (int)(((long long)mem_a_fp * (long long)ii + 32768LL) >> 16)
           + (int)(((long long)mem_b_fp * (long long)mi + 32768LL) >> 16);

        // 3. MIX INJECTION INTO SENSE
        sr = (int)(((long long)sense_a_fp * (long long)ir + 32768LL) >> 16)
           + (int)(((long long)sense_b_fp * (long long)sr + 32768LL) >> 16);
        si = (int)(((long long)sense_a_fp * (long long)ii + 32768LL) >> 16)
           + (int)(((long long)sense_b_fp * (long long)si + 32768LL) >> 16);

        // === ENERGY-TO-C: discord = z_sense - z_mem (before iterations) ===
        // Proportional controller: c = c_base + eta * discord
        // Low discord (balanced) -> c stays near base
        // High discord (divergent) -> c shifts to correct
        int discord_re = sr - mr;
        int discord_im = si - mi;
        int e_delta_re = (int)(((long long)energy_scale * (long long)discord_re + 32768LL) >> 16);
        int e_delta_im = (int)(((long long)energy_scale * (long long)discord_im + 32768LL) >> 16);

        // 4. JULIA ITERATIONS - MEMORY (c = c_base + energy_delta)
        int pre_mr = mr, pre_mi = mi;
        for (int li = 0; li < n_layers; li++) {
            int c_offset = li * cdim + tid;
            int cr = c_mem_re_base[c_offset] + e_delta_re;
            int ci = c_mem_im_base[c_offset] + e_delta_im;
            for (int it = 0; it < mem_iter; it++) {
                mr = ((mr + 131072) & 0x3FFFF) - 131072;
                mr = (int)(((long long)mr * 102944LL + 32768LL) >> 16);
                mi = ((mi + 131072) & 0x3FFFF) - 131072;
                mi = (int)(((long long)mi * 102944LL + 32768LL) >> 16);
                long long zr64 = (long long)mr;
                long long zi64 = (long long)mi;
                int re2  = (int)((zr64 * zr64 + 32768LL) >> 16);
                int im2  = (int)((zi64 * zi64 + 32768LL) >> 16);
                int reim = (int)((zr64 * zi64 + 32768LL) >> 16);
                mr = re2 - im2 + cr;
                mi = (reim << 1) + ci;
            }
        }
        mr = (int)(((long long)res_inv_fp * (long long)mr + 32768LL) >> 16)
           + (int)(((long long)res_fp * (long long)pre_mr + 32768LL) >> 16);
        mi = (int)(((long long)res_inv_fp * (long long)mi + 32768LL) >> 16)
           + (int)(((long long)res_fp * (long long)pre_mi + 32768LL) >> 16);

        // 5. JULIA ITERATIONS - SENSE (c = c_base + energy_delta)
        for (int li = 0; li < n_layers; li++) {
            int c_offset = li * cdim + tid;
            int cr = c_sense_re_base[c_offset] + e_delta_re;
            int ci = c_sense_im_base[c_offset] + e_delta_im;
            for (int it = 0; it < sense_iter; it++) {
                sr = ((sr + 131072) & 0x3FFFF) - 131072;
                sr = (int)(((long long)sr * 102944LL + 32768LL) >> 16);
                si = ((si + 131072) & 0x3FFFF) - 131072;
                si = (int)(((long long)si * 102944LL + 32768LL) >> 16);
                long long zr64 = (long long)sr;
                long long zi64 = (long long)si;
                int re2  = (int)((zr64 * zr64 + 32768LL) >> 16);
                int im2  = (int)((zi64 * zi64 + 32768LL) >> 16);
                int reim = (int)((zr64 * zi64 + 32768LL) >> 16);
                sr = re2 - im2 + cr;
                si = (reim << 1) + ci;
            }
            int lo_idx = ((bid * seq_len + t) * n_layers + li) * cdim + tid;
            layer_re_seq[lo_idx] = sr;
            layer_im_seq[lo_idx] = si;
        }

        // 6. OUTPUT = MEMORY x SENSE (complex multiply + torus fold)
        long long ar64 = (long long)mr, ai64 = (long long)mi;
        long long br64 = (long long)sr, bi64 = (long long)si;
        int o_re = (int)((ar64 * br64 + 32768LL) >> 16) - (int)((ai64 * bi64 + 32768LL) >> 16);
        int o_im = (int)((ar64 * bi64 + 32768LL) >> 16) + (int)((ai64 * br64 + 32768LL) >> 16);
        o_re = ((o_re + 131072) & 0x3FFFF) - 131072;
        o_re = (int)(((long long)o_re * 102944LL + 32768LL) >> 16);
        o_im = ((o_im + 131072) & 0x3FFFF) - 131072;
        o_im = (int)(((long long)o_im * 102944LL + 32768LL) >> 16);

        // Store output
        int out_idx = (bid * seq_len + t) * cdim + tid;
        out_re_seq[out_idx] = o_re;
        out_im_seq[out_idx] = o_im;

        // No post-iteration c update -- c is always freshly computed from base + discord
    }

    // Write back final state
    mem_re[state_idx] = mr;
    mem_im[state_idx] = mi;
    sense_re[state_idx] = sr;
    sense_im[state_idx] = si;
}
'''


# ===========================================================================
# PER-TOKEN C-VALUE FUSED KERNEL
# ===========================================================================
_PERTOKEN_C_KERNEL_SRC = r'''
extern "C" __global__
void reservoir_seq_pertoken_c(
    // --- State arrays (batch_size, cdim) int32, read+write ---
    int* __restrict__ mem_re,
    int* __restrict__ mem_im,
    int* __restrict__ sense_re,
    int* __restrict__ sense_im,

    // --- Injection sequence (batch_size, seq_len, cdim) int32, contiguous ---
    const int* __restrict__ inj_re_seq,
    const int* __restrict__ inj_im_seq,

    // --- Token IDs (batch_size, seq_len) int32 ---
    const int* __restrict__ token_ids,

    // --- Per-token C values: sense (vocab, n_layers, cdim) int32 ---
    const int* __restrict__ c_sense_re,
    const int* __restrict__ c_sense_im,

    // --- Per-token C values: memory (vocab, n_layers, cdim) int32 ---
    const int* __restrict__ c_mem_re,
    const int* __restrict__ c_mem_im,

    // --- Output sequence (batch_size, seq_len, cdim) int32 ---
    int* __restrict__ out_re_seq,
    int* __restrict__ out_im_seq,

    // --- Layer outputs (batch_size, seq_len, n_layers, cdim) int32 ---
    int* __restrict__ layer_re_seq,
    int* __restrict__ layer_im_seq,

    // --- Fixed-point mixing constants ---
    int decay_fp,
    int mem_a_fp, int mem_b_fp,
    int sense_a_fp, int sense_b_fp,
    int res_fp, int res_inv_fp,

    // --- Dimensions ---
    int cdim, int n_layers, int mem_iter, int sense_iter,
    int batch_size, int seq_len
) {
    int bid = blockIdx.x;
    int tid = blockIdx.y * blockDim.x + threadIdx.x;
    if (bid >= batch_size || tid >= cdim) return;

    int state_idx = bid * cdim + tid;

    // Load state into registers
    int mr = mem_re[state_idx];
    int mi = mem_im[state_idx];
    int sr = sense_re[state_idx];
    int si = sense_im[state_idx];

    // Loop over all tokens in sequence
    for (int t = 0; t < seq_len; t++) {
        int inj_idx = (bid * seq_len + t) * cdim + tid;
        int ir = inj_re_seq[inj_idx];
        int ii = inj_im_seq[inj_idx];

        // Look up per-token c-value base offset
        int tok = token_ids[bid * seq_len + t];
        int c_base = tok * n_layers * cdim;  // vocab-indexed

        // 1. SENSE DECAY
        sr = (int)(((long long)decay_fp * (long long)sr + 32768LL) >> 16);
        si = (int)(((long long)decay_fp * (long long)si + 32768LL) >> 16);

        // 2. MIX INJECTION INTO MEMORY
        mr = (int)(((long long)mem_a_fp * (long long)ir + 32768LL) >> 16)
           + (int)(((long long)mem_b_fp * (long long)mr + 32768LL) >> 16);
        mi = (int)(((long long)mem_a_fp * (long long)ii + 32768LL) >> 16)
           + (int)(((long long)mem_b_fp * (long long)mi + 32768LL) >> 16);

        // 3. MIX INJECTION INTO SENSE
        sr = (int)(((long long)sense_a_fp * (long long)ir + 32768LL) >> 16)
           + (int)(((long long)sense_b_fp * (long long)sr + 32768LL) >> 16);
        si = (int)(((long long)sense_a_fp * (long long)ii + 32768LL) >> 16)
           + (int)(((long long)sense_b_fp * (long long)si + 32768LL) >> 16);

        // 4. JULIA ITERATIONS - MEMORY (with residual)
        int pre_mr = mr, pre_mi = mi;
        for (int li = 0; li < n_layers; li++) {
            int c_offset = c_base + li * cdim + tid;
            int cr = c_mem_re[c_offset];
            int ci = c_mem_im[c_offset];
            for (int it = 0; it < mem_iter; it++) {
                mr = ((mr + 131072) & 0x3FFFF) - 131072;
                mr = (int)(((long long)mr * 102944LL + 32768LL) >> 16);
                mi = ((mi + 131072) & 0x3FFFF) - 131072;
                mi = (int)(((long long)mi * 102944LL + 32768LL) >> 16);
                long long zr64 = (long long)mr;
                long long zi64 = (long long)mi;
                int re2  = (int)((zr64 * zr64 + 32768LL) >> 16);
                int im2  = (int)((zi64 * zi64 + 32768LL) >> 16);
                int reim = (int)((zr64 * zi64 + 32768LL) >> 16);
                mr = re2 - im2 + cr;
                mi = (reim << 1) + ci;
            }
        }
        mr = (int)(((long long)res_inv_fp * (long long)mr + 32768LL) >> 16)
           + (int)(((long long)res_fp * (long long)pre_mr + 32768LL) >> 16);
        mi = (int)(((long long)res_inv_fp * (long long)mi + 32768LL) >> 16)
           + (int)(((long long)res_fp * (long long)pre_mi + 32768LL) >> 16);

        // 5. JULIA ITERATIONS - SENSE (capture per-layer)
        for (int li = 0; li < n_layers; li++) {
            int c_offset = c_base + li * cdim + tid;
            int cr = c_sense_re[c_offset];
            int ci = c_sense_im[c_offset];
            for (int it = 0; it < sense_iter; it++) {
                sr = ((sr + 131072) & 0x3FFFF) - 131072;
                sr = (int)(((long long)sr * 102944LL + 32768LL) >> 16);
                si = ((si + 131072) & 0x3FFFF) - 131072;
                si = (int)(((long long)si * 102944LL + 32768LL) >> 16);
                long long zr64 = (long long)sr;
                long long zi64 = (long long)si;
                int re2  = (int)((zr64 * zr64 + 32768LL) >> 16);
                int im2  = (int)((zi64 * zi64 + 32768LL) >> 16);
                int reim = (int)((zr64 * zi64 + 32768LL) >> 16);
                sr = re2 - im2 + cr;
                si = (reim << 1) + ci;
            }
            int lo_idx = ((bid * seq_len + t) * n_layers + li) * cdim + tid;
            layer_re_seq[lo_idx] = sr;
            layer_im_seq[lo_idx] = si;
        }

        // 6. OUTPUT = MEMORY x SENSE (complex multiply + torus fold)
        long long ar64 = (long long)mr, ai64 = (long long)mi;
        long long br64 = (long long)sr, bi64 = (long long)si;
        int o_re = (int)((ar64 * br64 + 32768LL) >> 16) - (int)((ai64 * bi64 + 32768LL) >> 16);
        int o_im = (int)((ar64 * bi64 + 32768LL) >> 16) + (int)((ai64 * br64 + 32768LL) >> 16);
        o_re = ((o_re + 131072) & 0x3FFFF) - 131072;
        o_re = (int)(((long long)o_re * 102944LL + 32768LL) >> 16);
        o_im = ((o_im + 131072) & 0x3FFFF) - 131072;
        o_im = (int)(((long long)o_im * 102944LL + 32768LL) >> 16);

        int out_idx = (bid * seq_len + t) * cdim + tid;
        out_re_seq[out_idx] = o_re;
        out_im_seq[out_idx] = o_im;
    }

    // Write back final state
    mem_re[state_idx] = mr;
    mem_im[state_idx] = mi;
    sense_re[state_idx] = sr;
    sense_im[state_idx] = si;
}
'''


# ===========================================================================
# FIXED-POINT CONSTANTS (must match jslm_v6.py exactly)
# ===========================================================================
FRAC = 16
ONE = 1 << FRAC       # 65536
HALF = 1 << (FRAC - 1)  # 32768


def _to_fp_scalar(x):
    """Convert a float to fixed-point int32 scalar."""
    return int(np.clip(np.round(x * ONE), -2**31, 2**31 - 1))


def _to_fp(arr):
    """Convert float array to fixed-point int32."""
    return np.clip(np.round(np.asarray(arr, np.float64) * ONE), -2**31, 2**31 - 1).astype(np.int32)


# ===========================================================================
# GPU DUAL RESERVOIR
# ===========================================================================

class GPUDualReservoir:
    """GPU-accelerated dual-state Julia reservoir using CuPy RawKernel.

    Mirrors the DualReservoir API from jslm_v6.py but processes a batch
    of sequences in parallel on the GPU.

    All state arrays are CuPy GPU arrays. The token loop stays in Python
    (sequential recurrence) but each step is one kernel launch.
    """

    def __init__(self, dim, c_values, n_iter=4,
                 mem_alpha=0.2, sense_alpha=0.8, sense_decay=0.1,
                 mem_iter=None, residual_strength=0.3,
                 jitter=0.0, mem_c_mag=0.95, iter_dropout=0.0,
                 trainable_c=False, vocab=0, c_mask_sparsity=0.0,
                 ptc_init=None, soft_fold=False, fold_scale=1.0):
        assert dim % 2 == 0
        self.dim = dim
        self.cdim = dim // 2
        self.n_iter = n_iter
        self.mem_iter = mem_iter if mem_iter is not None else max(1, n_iter // 2)
        self.mem_alpha = mem_alpha
        self.sense_alpha = sense_alpha
        self.sense_decay = sense_decay
        self.residual_strength = residual_strength
        self.jitter = jitter
        self.mem_c_mag = mem_c_mag
        self.iter_dropout = iter_dropout
        self.trainable_c = trainable_c
        self.soft_fold = soft_fold
        self.fold_scale = fold_scale
        self.c_adapt_rate = 0.0  # adaptive c feedback rate (0 = off)
        self.energy_feedback_rate = 0.0  # energy-to-c feedback rate (0 = off)
        self._training = True

        # Compile kernels with chosen fold mode
        self._k = _get_kernels(soft_fold, fold_scale)

        # Fixed-point mixing constants (scalars, passed to kernel)
        self._decay_fp = _to_fp_scalar(sense_decay)
        self._mem_a_fp = _to_fp_scalar(mem_alpha)
        self._mem_b_fp = _to_fp_scalar(1.0 - mem_alpha)
        self._sense_a_fp = _to_fp_scalar(sense_alpha)
        self._sense_b_fp = _to_fp_scalar(1.0 - sense_alpha)
        self._res_fp = _to_fp_scalar(residual_strength)
        self._res_inv_fp = _to_fp_scalar(1.0 - residual_strength)

        # Parse c values: list of complex or list of list of complex
        is_banked = len(c_values) > 0 and isinstance(c_values[0], (list, tuple))
        self.n_layers = len(c_values)

        # Build c arrays: (n_layers, cdim) int32 on GPU
        # For simplicity, we use bank index 0 (n_c_bank=1)
        sense_c_re = np.zeros((self.n_layers, self.cdim), dtype=np.int32)
        sense_c_im = np.zeros((self.n_layers, self.cdim), dtype=np.int32)
        mem_c_re = np.zeros((self.n_layers, self.cdim), dtype=np.int32)
        mem_c_im = np.zeros((self.n_layers, self.cdim), dtype=np.int32)

        # Float shadows for SPSA training
        self.c_re_float = np.zeros((self.n_layers, self.cdim), dtype=np.float32)
        self.c_im_float = np.zeros((self.n_layers, self.cdim), dtype=np.float32)
        self.mem_c_re_float = np.zeros((self.n_layers, self.cdim), dtype=np.float32)
        self.mem_c_im_float = np.zeros((self.n_layers, self.cdim), dtype=np.float32)

        var = np.linspace(-0.003, 0.003, self.cdim).astype(np.float32)

        for li, cv in enumerate(c_values):
            c = cv[0] if is_banked else cv
            # Sense c values
            crf = np.full(self.cdim, c.real, dtype=np.float32) + var
            cif = np.full(self.cdim, c.imag, dtype=np.float32) + var * 0.7
            self.c_re_float[li] = crf
            self.c_im_float[li] = cif
            sense_c_re[li] = _to_fp(crf)
            sense_c_im[li] = _to_fp(cif)
            # Memory c values (rotated, scaled)
            c_rot = c * complex(np.cos(0.5), np.sin(0.5)) * mem_c_mag
            mrf = np.full(self.cdim, c_rot.real, dtype=np.float32) + var * 1.2
            mif = np.full(self.cdim, c_rot.imag, dtype=np.float32) + var * 0.5
            self.mem_c_re_float[li] = mrf
            self.mem_c_im_float[li] = mif
            mem_c_re[li] = _to_fp(mrf)
            mem_c_im[li] = _to_fp(mif)

        # === C-VALUE SPARSE MASK ===
        # When sparsity > 0, only (1-sparsity) fraction of c dimensions are active.
        # Masked dims get c=0, so z = z^2 (pure squaring = chaotic for |z|>1).
        # This breaks cancellation and pushes LE positive.
        self.c_mask_sparsity = c_mask_sparsity
        if c_mask_sparsity > 0:
            mask_rng = np.random.RandomState(7)  # fixed seed for reproducibility
            self.c_mask = (mask_rng.rand(self.n_layers, self.cdim) >= c_mask_sparsity).astype(np.float32)
            n_active = int(self.c_mask.sum())
            n_total = self.n_layers * self.cdim
            print(f"  C-mask: {c_mask_sparsity:.0%} sparse → {n_active}/{n_total} active "
                  f"({n_active/n_total:.1%})")
            # Apply mask to shared c values
            for li in range(self.n_layers):
                self.c_re_float[li] *= self.c_mask[li]
                self.c_im_float[li] *= self.c_mask[li]
                self.mem_c_re_float[li] *= self.c_mask[li]
                self.mem_c_im_float[li] *= self.c_mask[li]
                sense_c_re[li] = _to_fp(self.c_re_float[li])
                sense_c_im[li] = _to_fp(self.c_im_float[li])
                mem_c_re[li] = _to_fp(self.mem_c_re_float[li])
                mem_c_im[li] = _to_fp(self.mem_c_im_float[li])
        else:
            self.c_mask = None

        # Upload shared c values to GPU
        self._c_sense_re = cp.asarray(sense_c_re)
        self._c_sense_im = cp.asarray(sense_c_im)
        self._c_mem_re = cp.asarray(mem_c_re)
        self._c_mem_im = cp.asarray(mem_c_im)

        # === PER-TOKEN C VALUES ===
        self.vocab = vocab
        self.per_token_c = vocab > 0
        if self.per_token_c:
            if ptc_init is not None:
                # Use Mandelbrot-structured initialization
                self.ptc_sense_re, self.ptc_sense_im, \
                    self.ptc_mem_re, self.ptc_mem_im = ptc_init
                print(f"  Per-token c: {vocab} × {self.n_layers} × {self.cdim} = "
                      f"{vocab * self.n_layers * self.cdim * 4:,} params "
                      f"(Mandelbrot-structured init)")
            else:
                # Default: tile shared c + noise
                self.ptc_sense_re = np.tile(self.c_re_float, (vocab, 1, 1))  # (V, L, C)
                self.ptc_sense_im = np.tile(self.c_im_float, (vocab, 1, 1))
                self.ptc_mem_re = np.tile(self.mem_c_re_float, (vocab, 1, 1))
                self.ptc_mem_im = np.tile(self.mem_c_im_float, (vocab, 1, 1))
                # Add per-token variation so they're not all identical
                rng = np.random.RandomState(42)
                for arr in [self.ptc_sense_re, self.ptc_sense_im,
                            self.ptc_mem_re, self.ptc_mem_im]:
                    arr += rng.randn(*arr.shape).astype(np.float32) * 0.005
                print(f"  Per-token c: {vocab} × {self.n_layers} × {self.cdim} = "
                      f"{vocab * self.n_layers * self.cdim * 4:,} params")
            self._sync_ptc_to_gpu()

        # Adam state for trainable c
        if trainable_c:
            if self.per_token_c:
                n_c = vocab * self.n_layers * self.cdim * 4
            else:
                n_c = self.n_layers * self.cdim * 4
            self._c_adam_m = np.zeros(n_c, dtype=np.float32)
            self._c_adam_v = np.zeros(n_c, dtype=np.float32)
            self._c_step = 0

        # Batch state (allocated on first reset)
        self._batch_size = 0
        self._mem_re = None
        self._mem_im = None
        self._sense_re = None
        self._sense_im = None

    def _sync_ptc_to_gpu(self):
        """Upload per-token c values to GPU as (vocab, n_layers, cdim) int32."""
        # Apply sparse mask before upload (mask is (n_layers, cdim), broadcasts over vocab)
        sre = self.ptc_sense_re
        sim = self.ptc_sense_im
        mre = self.ptc_mem_re
        mim = self.ptc_mem_im
        if self.c_mask is not None:
            sre = sre * self.c_mask[np.newaxis, :, :]
            sim = sim * self.c_mask[np.newaxis, :, :]
            mre = mre * self.c_mask[np.newaxis, :, :]
            mim = mim * self.c_mask[np.newaxis, :, :]
        self._ptc_sense_re = cp.asarray(_to_fp(sre.reshape(-1)).reshape(
            self.vocab, self.n_layers, self.cdim))
        self._ptc_sense_im = cp.asarray(_to_fp(sim.reshape(-1)).reshape(
            self.vocab, self.n_layers, self.cdim))
        self._ptc_mem_re = cp.asarray(_to_fp(mre.reshape(-1)).reshape(
            self.vocab, self.n_layers, self.cdim))
        self._ptc_mem_im = cp.asarray(_to_fp(mim.reshape(-1)).reshape(
            self.vocab, self.n_layers, self.cdim))

    def reset(self, batch_size):
        """Zero all state arrays for a new batch (reuse buffers if same size)."""
        self._batch_size = batch_size
        if (hasattr(self, '_mem_re') and self._mem_re is not None
                and self._mem_re.shape[0] == batch_size):
            self._mem_re.fill(0)
            self._mem_im.fill(0)
            self._sense_re.fill(0)
            self._sense_im.fill(0)
        else:
            self._mem_re = cp.zeros((batch_size, self.cdim), dtype=cp.int32)
            self._mem_im = cp.zeros((batch_size, self.cdim), dtype=cp.int32)
            self._sense_re = cp.zeros((batch_size, self.cdim), dtype=cp.int32)
            self._sense_im = cp.zeros((batch_size, self.cdim), dtype=cp.int32)

    def _sync_c_to_gpu(self):
        """Recompute int32 c values from float shadows and upload to GPU."""
        sense_c_re = np.zeros((self.n_layers, self.cdim), dtype=np.int32)
        sense_c_im = np.zeros((self.n_layers, self.cdim), dtype=np.int32)
        mem_c_re = np.zeros((self.n_layers, self.cdim), dtype=np.int32)
        mem_c_im = np.zeros((self.n_layers, self.cdim), dtype=np.int32)
        for li in range(self.n_layers):
            crf = self.c_re_float[li]
            cif = self.c_im_float[li]
            mrf = self.mem_c_re_float[li]
            mif = self.mem_c_im_float[li]
            if self.c_mask is not None:
                crf = crf * self.c_mask[li]
                cif = cif * self.c_mask[li]
                mrf = mrf * self.c_mask[li]
                mif = mif * self.c_mask[li]
            sense_c_re[li] = _to_fp(crf)
            sense_c_im[li] = _to_fp(cif)
            mem_c_re[li] = _to_fp(mrf)
            mem_c_im[li] = _to_fp(mif)
        self._c_sense_re = cp.asarray(sense_c_re)
        self._c_sense_im = cp.asarray(sense_c_im)
        self._c_mem_re = cp.asarray(mem_c_re)
        self._c_mem_im = cp.asarray(mem_c_im)

    def get_c_flat(self):
        """Flatten all c values into a single float32 vector for SPSA."""
        if self.per_token_c:
            return np.concatenate([
                self.ptc_sense_re.ravel(), self.ptc_sense_im.ravel(),
                self.ptc_mem_re.ravel(), self.ptc_mem_im.ravel(),
            ])
        parts = []
        for li in range(self.n_layers):
            parts.extend([
                self.c_re_float[li],
                self.c_im_float[li],
                self.mem_c_re_float[li],
                self.mem_c_im_float[li],
            ])
        return np.concatenate(parts)

    def set_c_flat(self, flat):
        """Set all c values from a flat float32 vector and re-upload to GPU."""
        if self.per_token_c:
            chunk = self.vocab * self.n_layers * self.cdim
            self.ptc_sense_re = flat[0:chunk].reshape(self.vocab, self.n_layers, self.cdim).copy()
            self.ptc_sense_im = flat[chunk:2*chunk].reshape(self.vocab, self.n_layers, self.cdim).copy()
            self.ptc_mem_re = flat[2*chunk:3*chunk].reshape(self.vocab, self.n_layers, self.cdim).copy()
            self.ptc_mem_im = flat[3*chunk:4*chunk].reshape(self.vocab, self.n_layers, self.cdim).copy()
            self._sync_ptc_to_gpu()
            return
        offset = 0
        for li in range(self.n_layers):
            self.c_re_float[li] = flat[offset:offset + self.cdim].copy()
            offset += self.cdim
            self.c_im_float[li] = flat[offset:offset + self.cdim].copy()
            offset += self.cdim
            self.mem_c_re_float[li] = flat[offset:offset + self.cdim].copy()
            offset += self.cdim
            self.mem_c_im_float[li] = flat[offset:offset + self.cdim].copy()
            offset += self.cdim
        self._sync_c_to_gpu()

    def step_batch(self, inj_re_gpu, inj_im_gpu):
        """One token step for the whole batch. Returns (out_re, out_im, layer_re, layer_im).

        Args:
            inj_re_gpu: (batch_size, cdim) int32 CuPy array — injection real part
            inj_im_gpu: (batch_size, cdim) int32 CuPy array — injection imag part

        Returns:
            out_re: (batch_size, cdim) int32 — interference output
            out_im: (batch_size, cdim) int32
            layer_re: (batch_size, n_layers, cdim) int32 — per-layer sense states
            layer_im: (batch_size, n_layers, cdim) int32
        """
        bs = self._batch_size
        cdim = self.cdim

        # Allocate output buffers
        out_re = cp.zeros((bs, cdim), dtype=cp.int32)
        out_im = cp.zeros((bs, cdim), dtype=cp.int32)
        layer_re = cp.zeros((bs, self.n_layers, cdim), dtype=cp.int32)
        layer_im = cp.zeros((bs, self.n_layers, cdim), dtype=cp.int32)

        # Determine iteration counts (handle iter_dropout)
        sense_iter = self.n_iter
        if (self.iter_dropout > 0 and self._training and
                sense_iter > 1 and np.random.random() < self.iter_dropout):
            sense_iter -= 1

        # Launch kernel: grid=(batch, cdim_blocks), block=(THREADS,)
        THREADS = 256
        cdim_blocks = (cdim + THREADS - 1) // THREADS
        self._k['step'](
            (bs, cdim_blocks), (THREADS,),
            (
                self._mem_re, self._mem_im,
                self._sense_re, self._sense_im,
                inj_re_gpu, inj_im_gpu,
                self._c_sense_re, self._c_sense_im,
                self._c_mem_re, self._c_mem_im,
                out_re, out_im,
                layer_re, layer_im,
                np.int32(self._decay_fp),
                np.int32(self._mem_a_fp),
                np.int32(self._mem_b_fp),
                np.int32(self._sense_a_fp),
                np.int32(self._sense_b_fp),
                np.int32(self._res_fp),
                np.int32(self._res_inv_fp),
                np.int32(cdim),
                np.int32(self.n_layers),
                np.int32(self.mem_iter),
                np.int32(sense_iter),
                np.int32(bs),
            )
        )

        return out_re, out_im, layer_re, layer_im

    def forward_seq_batch(self, inj_re_seq, inj_im_seq, token_ids_gpu=None):
        """Run a full sequence through the reservoir for a batch.

        Args:
            inj_re_seq: (batch_size, seq_len, cdim) int32 CuPy array
            inj_im_seq: (batch_size, seq_len, cdim) int32 CuPy array
            token_ids_gpu: (batch_size, seq_len) int32 CuPy array — needed for per-token c

        Returns:
            out_re: (batch_size, seq_len, cdim) int32
            out_im: (batch_size, seq_len, cdim) int32
            layer_re: (batch_size, seq_len, n_layers, cdim) int32
            layer_im: (batch_size, seq_len, n_layers, cdim) int32
        """
        bs, seq_len, cdim = inj_re_seq.shape

        # Allocate sequence output buffers
        all_out_re = cp.zeros((bs, seq_len, cdim), dtype=cp.int32)
        all_out_im = cp.zeros((bs, seq_len, cdim), dtype=cp.int32)
        all_layer_re = cp.zeros((bs, seq_len, self.n_layers, cdim), dtype=cp.int32)
        all_layer_im = cp.zeros((bs, seq_len, self.n_layers, cdim), dtype=cp.int32)

        # Use fused kernel if injection is contiguous (avoids Python loop)
        is_contig = inj_re_seq.flags['C_CONTIGUOUS'] and inj_im_seq.flags['C_CONTIGUOUS']

        # Energy-to-C path: c = c_base + η * (z_sense - z_mem) — self-correcting manifold
        if self.energy_feedback_rate != 0.0 and is_contig:
            return self._forward_fused_energy(inj_re_seq, inj_im_seq, bs, seq_len, cdim,
                                               all_out_re, all_out_im, all_layer_re, all_layer_im)

        # Adaptive c path: landscape evolves per-token via output feedback
        if self.c_adapt_rate != 0.0 and is_contig:
            return self._forward_fused_adaptive(inj_re_seq, inj_im_seq, bs, seq_len, cdim,
                                                 all_out_re, all_out_im, all_layer_re, all_layer_im)

        # Per-token c path
        if self.per_token_c and token_ids_gpu is not None and is_contig:
            return self._forward_fused_pertoken(
                inj_re_seq, inj_im_seq, token_ids_gpu, bs, seq_len, cdim,
                all_out_re, all_out_im, all_layer_re, all_layer_im)

        # Shared c path
        if is_contig:
            return self._forward_fused(inj_re_seq, inj_im_seq, bs, seq_len, cdim,
                                       all_out_re, all_out_im, all_layer_re, all_layer_im)

        # Fallback: sequential token loop
        for t in range(seq_len):
            o_re, o_im, l_re, l_im = self.step_batch(
                cp.ascontiguousarray(inj_re_seq[:, t, :]),
                cp.ascontiguousarray(inj_im_seq[:, t, :]))
            all_out_re[:, t, :] = o_re
            all_out_im[:, t, :] = o_im
            all_layer_re[:, t, :, :] = l_re
            all_layer_im[:, t, :, :] = l_im

        return all_out_re, all_out_im, all_layer_re, all_layer_im

    def _forward_fused_pertoken(self, inj_re_seq, inj_im_seq, token_ids_gpu,
                                bs, seq_len, cdim, out_re, out_im, layer_re, layer_im):
        """Single kernel launch with per-token c values."""
        sense_iter = self.n_iter
        THREADS = 256
        cdim_blocks = (cdim + THREADS - 1) // THREADS
        self._k['pertoken'](
            (bs, cdim_blocks), (THREADS,),
            (self._mem_re, self._mem_im, self._sense_re, self._sense_im,
             inj_re_seq, inj_im_seq,
             token_ids_gpu,
             self._ptc_sense_re, self._ptc_sense_im,
             self._ptc_mem_re, self._ptc_mem_im,
             out_re, out_im, layer_re, layer_im,
             np.int32(self._decay_fp),
             np.int32(self._mem_a_fp), np.int32(self._mem_b_fp),
             np.int32(self._sense_a_fp), np.int32(self._sense_b_fp),
             np.int32(self._res_fp), np.int32(self._res_inv_fp),
             np.int32(cdim), np.int32(self.n_layers),
             np.int32(self.mem_iter), np.int32(sense_iter),
             np.int32(bs), np.int32(seq_len))
        )
        return out_re, out_im, layer_re, layer_im

    def _forward_fused(self, inj_re_seq, inj_im_seq, bs, seq_len, cdim,
                       out_re, out_im, layer_re, layer_im):
        """Single kernel launch for entire sequence."""
        sense_iter = self.n_iter
        THREADS = 256
        cdim_blocks = (cdim + THREADS - 1) // THREADS
        self._k['fused'](
            (bs, cdim_blocks), (THREADS,),
            (self._mem_re, self._mem_im, self._sense_re, self._sense_im,
             inj_re_seq, inj_im_seq,
             self._c_sense_re, self._c_sense_im,
             self._c_mem_re, self._c_mem_im,
             out_re, out_im, layer_re, layer_im,
             np.int32(self._decay_fp),
             np.int32(self._mem_a_fp), np.int32(self._mem_b_fp),
             np.int32(self._sense_a_fp), np.int32(self._sense_b_fp),
             np.int32(self._res_fp), np.int32(self._res_inv_fp),
             np.int32(cdim), np.int32(self.n_layers),
             np.int32(self.mem_iter), np.int32(sense_iter),
             np.int32(bs), np.int32(seq_len))
        )
        return out_re, out_im, layer_re, layer_im

    def _forward_fused_adaptive(self, inj_re_seq, inj_im_seq, bs, seq_len, cdim,
                                 out_re, out_im, layer_re, layer_im):
        """Single kernel launch with adaptive c feedback — landscape evolves per-token."""
        sense_iter = self.n_iter
        THREADS = 256
        cdim_blocks = (cdim + THREADS - 1) // THREADS

        # Allocate per-batch c arrays: each batch element gets its own copy
        # Initialized from shared base c values
        c_sense_re_batch = cp.broadcast_to(
            self._c_sense_re[None, :, :], (bs, self.n_layers, cdim)).copy()
        c_sense_im_batch = cp.broadcast_to(
            self._c_sense_im[None, :, :], (bs, self.n_layers, cdim)).copy()
        c_mem_re_batch = cp.broadcast_to(
            self._c_mem_re[None, :, :], (bs, self.n_layers, cdim)).copy()
        c_mem_im_batch = cp.broadcast_to(
            self._c_mem_im[None, :, :], (bs, self.n_layers, cdim)).copy()

        # Convert adapt rate to fixed-point
        adapt_fp = int(self.c_adapt_rate * 65536)

        self._k['adaptive'](
            (bs, cdim_blocks), (THREADS,),
            (self._mem_re, self._mem_im, self._sense_re, self._sense_im,
             inj_re_seq, inj_im_seq,
             c_sense_re_batch, c_sense_im_batch,
             c_mem_re_batch, c_mem_im_batch,
             out_re, out_im, layer_re, layer_im,
             np.int32(self._decay_fp),
             np.int32(self._mem_a_fp), np.int32(self._mem_b_fp),
             np.int32(self._sense_a_fp), np.int32(self._sense_b_fp),
             np.int32(self._res_fp), np.int32(self._res_inv_fp),
             np.int32(adapt_fp),
             np.int32(cdim), np.int32(self.n_layers),
             np.int32(self.mem_iter), np.int32(sense_iter),
             np.int32(bs), np.int32(seq_len))
        )
        return out_re, out_im, layer_re, layer_im

    def _forward_fused_energy(self, inj_re_seq, inj_im_seq, bs, seq_len, cdim,
                               out_re, out_im, layer_re, layer_im):
        """Energy-to-C kernel: c = c_base + η * (z_sense - z_mem).

        No per-batch c copies needed — uses shared base c arrays (read-only).
        Discord computed per-token before iterations; proportional, never drifts.
        """
        sense_iter = self.n_iter
        THREADS = 256
        cdim_blocks = (cdim + THREADS - 1) // THREADS

        energy_fp = int(self.energy_feedback_rate * 65536)

        self._k['energy'](
            (bs, cdim_blocks), (THREADS,),
            (self._mem_re, self._mem_im, self._sense_re, self._sense_im,
             inj_re_seq, inj_im_seq,
             self._c_sense_re, self._c_sense_im,
             self._c_mem_re, self._c_mem_im,
             out_re, out_im, layer_re, layer_im,
             np.int32(self._decay_fp),
             np.int32(self._mem_a_fp), np.int32(self._mem_b_fp),
             np.int32(self._sense_a_fp), np.int32(self._sense_b_fp),
             np.int32(self._res_fp), np.int32(self._res_inv_fp),
             np.int32(energy_fp),
             np.int32(cdim), np.int32(self.n_layers),
             np.int32(self.mem_iter), np.int32(sense_iter),
             np.int32(bs), np.int32(seq_len))
        )
        return out_re, out_im, layer_re, layer_im


# ===========================================================================
# INJECTION HELPERS (GPU)
# ===========================================================================

def embed_to_complex_gpu(embed_float, boost=3.0, gain=None, pi_phase=None):
    """Convert embedding matrix to complex injection (GPU version).

    Args:
        embed_float: (batch, seq, dim) or (seq, dim) float32 CuPy array
        boost: injection boost factor
        gain: (cdim,) float32 CuPy array or None
        pi_phase: (batch, seq) or (seq,) float32 CuPy array of π/2 * token_id, or None

    Returns:
        inj_re, inj_im: float32 CuPy arrays, same leading shape + (cdim,)
    """
    re = embed_float[..., 0::2]
    im = embed_float[..., 1::2]
    r = cp.sqrt(re ** 2 + im ** 2 + 1e-8)
    theta = cp.arctan2(im, re)
    # Per-token phase injection: unique irrational angular position per token
    if pi_phase is not None:
        theta = theta + pi_phase[..., cp.newaxis]
    r_b = r * boost
    if gain is not None:
        # Broadcast gain over batch/seq dims
        r_b = r_b * gain
    r_norm = r_b / (1.0 + r_b)
    return (r_norm * cp.cos(theta)).astype(cp.float32), \
           (r_norm * cp.sin(theta)).astype(cp.float32)


def to_fp_gpu(x):
    """Convert float CuPy array to fixed-point int32."""
    return cp.clip(cp.round(x.astype(cp.float64) * ONE), -2**31, 2**31 - 1).astype(cp.int32)


# ===========================================================================
# BATCH STATE EXTRACTION
# ===========================================================================

def prepare_injection_batch(model, token_ids_batch, device_id=None):
    """Prepare injection arrays for a batch of sequences on GPU.

    Args:
        model: JSLM instance (for embed, pos, gain, boost)
        token_ids_batch: list of numpy arrays (token id sequences)
            All must have the same length (pad beforehand if needed).

    Returns:
        inj_re: (batch_size, seq_len, cdim) int32 CuPy array
        inj_im: (batch_size, seq_len, cdim) int32 CuPy array
    """
    batch_size = len(token_ids_batch)
    seq_len = len(token_ids_batch[0])
    dim = model.dim

    # Cache embed/pos tables on GPU (first call uploads, subsequent reuse)
    if not hasattr(model, '_gpu_embed'):
        model._gpu_embed = cp.asarray(model.embed)
        model._gpu_pos = cp.asarray(model.pos)
        model._gpu_gain = cp.asarray(model.gain) if model.gain is not None else None
    else:
        # Sync gain every call — gain is updated by train loop (gain_decay, gap feedback)
        if model.gain is not None:
            model._gpu_gain[:] = cp.asarray(model.gain)

    # Upload token IDs to GPU and do lookup there
    if isinstance(token_ids_batch, np.ndarray):
        ids_gpu = cp.asarray(token_ids_batch.astype(np.int32, copy=False))
    else:
        ids_gpu = cp.asarray(np.array(token_ids_batch, dtype=np.int32))
    pos_idx = cp.clip(cp.arange(seq_len), 0, model.max_seq - 1)

    # GPU-side embedding + positional: (batch, seq, dim)
    embeds_gpu = model._gpu_embed[ids_gpu] + model._gpu_pos[pos_idx]

    # Per-token phase: π/2 * token_id (unique irrational angular position)
    pi_ph = None
    if getattr(model, 'pi_phase', False):
        pi_ph = cp.float32(np.pi / 2) * ids_gpu.astype(cp.float32)

    # Convert to complex injection
    inj_re_f, inj_im_f = embed_to_complex_gpu(embeds_gpu, model.boost, model._gpu_gain, pi_phase=pi_ph)

    # Convert to fixed-point
    inj_re = to_fp_gpu(inj_re_f)
    inj_im = to_fp_gpu(inj_im_f)

    return inj_re, inj_im


def add_jitter_gpu(inj_re, inj_im, jitter_scale):
    """Add random jitter noise to injection (in-place on GPU)."""
    if jitter_scale <= 0:
        return inj_re, inj_im
    noise_scale = int(jitter_scale * ONE)
    shape = inj_re.shape
    inj_re = inj_re + cp.random.randint(-noise_scale, noise_scale + 1,
                                          shape, dtype=cp.int32)
    inj_im = inj_im + cp.random.randint(-noise_scale, noise_scale + 1,
                                          shape, dtype=cp.int32)
    return inj_re, inj_im


def extract_features_gpu(out_re, out_im, layer_re, layer_im, layer_concat=True):
    """Convert int32 reservoir output to float32 features on GPU.

    FEATURE FOLDING: layers are folded into base dim via complex modulation.
    Output is always (batch, seq, dim) regardless of layer_concat.
    """
    bs, seq_len, cdim = out_re.shape
    dim = cdim * 2
    inv_one = cp.float32(1.0 / ONE)

    # Base interference features — use empty (we overwrite all entries)
    base = cp.empty((bs, seq_len, dim), dtype=cp.float32)
    base[:, :, 0::2] = out_re.astype(cp.float32) * inv_one
    base[:, :, 1::2] = out_im.astype(cp.float32) * inv_one

    if not layer_concat:
        return base

    # Fold each layer via complex modulation on GPU
    # (sequential: each layer modulates accumulated base)
    n_layers = layer_re.shape[2]
    for li in range(n_layers):
        # Convert layer directly — no intermediate allocation
        m_re = layer_re[:, :, li, :].astype(cp.float32) * inv_one
        m_im = layer_im[:, :, li, :].astype(cp.float32) * inv_one

        b_re = base[:, :, 0::2]
        b_im = base[:, :, 1::2]

        new_re = b_re * m_re - b_im * m_im
        new_im = b_re * m_im + b_im * m_re

        # Normalize to preserve magnitude
        norms = cp.sqrt(cp.sum(new_re**2 + new_im**2, axis=2, keepdims=True) + 1e-8)
        scale = cp.sqrt(cp.sum(b_re**2 + b_im**2, axis=2, keepdims=True) + 1e-8)
        ratio = scale / norms

        base[:, :, 0::2] = new_re * ratio
        base[:, :, 1::2] = new_im * ratio

    return base


def run_gpu_reservoir(model, token_ids_batch, gpu_res):
    """Full pipeline: prepare injection → run reservoir → extract features.

    Args:
        model: JSLM instance
        token_ids_batch: list of numpy arrays (same-length token sequences)
        gpu_res: GPUDualReservoir instance

    Returns:
        features: (batch, seq, feature_dim) float32 CuPy array
    """
    batch_size = len(token_ids_batch)
    seq_len = len(token_ids_batch[0])

    # 1. Prepare injection
    inj_re, inj_im = prepare_injection_batch(model, token_ids_batch)

    # 2. Add jitter during training
    if gpu_res._training and gpu_res.jitter > 0:
        inj_re, inj_im = add_jitter_gpu(inj_re, inj_im, gpu_res.jitter)

    # 3. Reset and run reservoir
    gpu_res.reset(batch_size)
    out_re, out_im, layer_re, layer_im = gpu_res.forward_seq_batch(inj_re, inj_im)

    # 4. Extract features
    features = extract_features_gpu(
        out_re, out_im, layer_re, layer_im,
        layer_concat=model.layer_concat)

    return features
